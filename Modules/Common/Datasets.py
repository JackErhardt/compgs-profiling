import os
import struct
from collections import namedtuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .DataStructure import Sample, DatasetSample
from .Utils import CustomLogger
from .AuxiliaryFunctions import load_point_clouds

from projectaria_tools.core import data_provider, image, calibration
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
from projectaria_tools.core.mps import read_closed_loop_trajectory, read_global_point_cloud, StreamCompressionMode
from projectaria_tools.projects.ase import get_ase_rgb_calibration

CameraModel = namedtuple(typename='CameraModel', field_names=['model_id', 'model_name', 'num_params'])
CAMERA_MODELS = {CameraModel(model_id=0, model_name='SIMPLE_PINHOLE', num_params=3), CameraModel(model_id=1, model_name='PINHOLE', num_params=4)}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


class BaseDataset(Dataset):
    def __init__(self, root: str, image_folder: str, logger: CustomLogger, device: str,  eval_interval: int = 8, z_near: float = 0.01, z_far: float = 100) -> None:
        """
        Dataset for loading images and camera calibration matrices.
        :param root: path to the root folder of the dataset.
        :param image_folder: folder that contains images.
        :param logger: logger to record information.
        :param device: device to load data.
        :param eval_interval: interval of evaluated views, default is 8. If set to -1, evaluation will be performed on the training set (i.e., the training and evaluation sets are identical).
        :param z_near: distance between camera and near clipping plane, default is 0.01.
        :param z_far: distance between camera and far clipping plane, default is 100.
        """
        self.root, self.logger, self.device, self.z_near, self.z_far = root, logger, device, z_near, z_far
        if not hasattr(self, 'image_folder'):
            self.image_folder = os.path.join(root, image_folder)
        self.logger.info('\nLoading data from {}...'.format(root))

        self.sfm_point_cloud_path = os.path.join(root, 'sparse', '0', 'points3D.bin')
        if os.path.exists(self.sfm_point_cloud_path):
            self.point_cloud = load_point_clouds(self.sfm_point_cloud_path)
        else:
            self.point_cloud = None

        # load camera calibration information
        cam_mat_folder = os.path.join(root, 'sparse', '0')
        img_folder = os.path.join(root, image_folder)
        samples = sorted(self.load_samples(cam_mat_folder=cam_mat_folder, img_folder=img_folder), key=lambda x: x.img_name)

        self.train_samples = {}  # training samples
        self.test_samples = {}  # test samples

        # train / eval split
        # If eval_interval is -1, evaluate on the training set (use all samples for both training and evaluation)
        if eval_interval == -1:
            self.train_samples = {idx: sample for idx, sample in enumerate(samples)}
            self.test_samples = dict(self.train_samples)
        else:
            self.train_samples = {idx: sample for idx, sample in enumerate(samples) if idx % eval_interval != 0}
            self.test_samples = {idx: sample for idx, sample in enumerate(samples) if idx % eval_interval == 0}

        # calculate the radius of the sphere that contains all training camera centers
        self.screen_extent = self.calculate_camera_sphere_radius()

        self.train_views_stack = list(self.train_samples.keys())  # stack to save views for training

        self.training = True  # flag to indicate if the dataset is in training mode

        logger.info(f'Loaded {len(self.train_samples)} training views and {len(self.test_samples)} evaluation views.')

    def __getitem__(self, idx: int) -> Sample:
        """
        Return image, intrinsic matrix and extrinsic matrix of the given index.
        :param idx: order of the data.
        :return sample: samples including image, camera_center, world_to_view_proj_mat, world_to_image_proj_mat, etc.
        """
        if not self.train_views_stack:  # re-create the training views stack
            self.logger.info('\nRe-creating training views stack...')
            self.train_views_stack = list(self.train_samples.keys())
        if self.training:  # randomly sample a view from the training views stack
            sample = self.train_samples[self.train_views_stack.pop(np.random.randint(0, len(self.train_views_stack)))]
        else:  # select a view from the test views
            sample = self.test_samples[list(self.test_samples.keys())[idx]]
        
        img = sample.img
        alpha_mask = sample.alpha_mask
        if img is None:
            img_path = os.path.join(self.image_folder, sample.img_name)
            pil_image = Image.open(img_path)
            img_np = np.array(pil_image)
            
            if img_np.shape[2] == 4:
                alpha_mask = torch.tensor(img_np[:, :, 3:4].transpose(2, 0, 1) / 255., dtype=torch.float, device=self.device)
                img = torch.tensor(img_np[:, :, :3].transpose(2, 0, 1) / 255., dtype=torch.float, device=self.device).clamp_(min=0., max=1.)
            else:
                img = torch.tensor(img_np.transpose(2, 0, 1) / 255., dtype=torch.float, device=self.device).clamp_(min=0., max=1.)
                alpha_mask = torch.ones((1, sample.img_height, sample.img_width), dtype=torch.float, device=self.device)

        sample = Sample(img=img, alpha_mask=alpha_mask, image_height=sample.img_height, image_width=sample.img_width,
                        tan_half_fov_x=sample.tan_half_fov_x, tan_half_fov_y=sample.tan_half_fov_y,
                        camera_center=sample.camera_center, cam_idx=sample.cam_idx, screen_extent=self.screen_extent,
                        world_to_view_proj_mat=sample.world_to_view_proj_mat,
                        world_to_image_proj_mat=sample.world_to_image_proj_mat,
                        img_name=sample.img_name)
        return sample

    def __len__(self) -> int:
        return len(self.train_samples) if self.training else len(self.test_samples)

    def eval(self) -> None:
        """
        Set the dataset to evaluation mode.
        """
        self.training = False

    def train(self) -> None:
        """
        Set the dataset to training mode.
        """
        self.training = True

    def load_samples(self, cam_mat_folder: str, img_folder: str) -> list[DatasetSample]:
        """
        Load camera calibration information and images.
        :param cam_mat_folder: path to the folder that contains camera calibration files.
        :param img_folder: path to the folder that contains images.
        """
        # load extrinsic data
        extrinsic_filepath = os.path.join(cam_mat_folder, 'images.bin')
        extrinsic_data = self.read_extrinsic_binary(extrinsic_filepath)

        img_names = {view_idx: extrinsic_data[view_idx]['img_name'] for view_idx in extrinsic_data}
        
        rotation_mats = {view_idx: extrinsic_data[view_idx]['rotation_mat'] for view_idx in extrinsic_data}
        translation_vectors = {view_idx: extrinsic_data[view_idx]['translation_vector'] for view_idx in extrinsic_data}

        # load intrinsic data
        intrinsic_filepath = os.path.join(cam_mat_folder, 'cameras.bin')
        intrinsic_data = self.read_intrinsic_binary(intrinsic_filepath)

        fov_x = {view_idx: intrinsic_data[extrinsic_data[view_idx]['camera_model_id']]['fov_x'] for view_idx in extrinsic_data}
        fov_y = {view_idx: intrinsic_data[extrinsic_data[view_idx]['camera_model_id']]['fov_y'] for view_idx in extrinsic_data}

        height = {view_idx: intrinsic_data[extrinsic_data[view_idx]['camera_model_id']]['height'] for view_idx in extrinsic_data}
        width = {view_idx: intrinsic_data[extrinsic_data[view_idx]['camera_model_id']]['width'] for view_idx in extrinsic_data}

        # calculate projection matrices and camera centers
        proj_matrices = {view_idx: self.calculate_proj_mats(
            rotation_mat=rotation_mats[view_idx], translation_vector=translation_vectors[view_idx], fov_x=fov_x[view_idx], fov_y=fov_y[view_idx]) for view_idx in extrinsic_data}

        cams_calib_info = [
            DatasetSample(
                cam_idx=view_idx, camera_center=proj_matrices[view_idx]['camera_center'],
                img_name=img_names[view_idx], img=None, img_height=height[view_idx], img_width=width[view_idx],
                rotation_mat=rotation_mats[view_idx], translation_vec=translation_vectors[view_idx],
                fov_x=fov_x[view_idx], fov_y=fov_y[view_idx], tan_half_fov_x=np.tan(fov_x[view_idx] / 2), tan_half_fov_y=np.tan(fov_y[view_idx] / 2),
                world_to_view_proj_mat=proj_matrices[view_idx]['world_to_view_proj_mat'],
                world_to_image_proj_mat=proj_matrices[view_idx]['world_to_image_proj_mat'],
                perspective_proj_mat=proj_matrices[view_idx]['perspective_proj_mat'])
            for view_idx in extrinsic_data]
        return cams_calib_info

    def calculate_proj_mats(self, rotation_mat: np.ndarray, translation_vector: np.ndarray, fov_x: float, fov_y: float) -> dict:
        """
        Calculate projection matrices and camera centers.
        :param rotation_mat: rotation matrix with shape (3, 3).
        :param translation_vector: translation vector with shape (3,).
        :param fov_x: field of view in x direction.
        :param fov_y: field of view in y direction.
        :return world_to_view_proj_mat: world to view projection matrix with shape (4, 4).
        :return perspective_proj_mat: perspective projection matrix with shape (4, 4).
        :return world_to_image_proj_mat: world to image projection matrix with shape (4, 4).
        :return camera_center: camera center in world space with shape (3,).
        """
        # calculate world to view projection matrices
        world_to_view_proj_mat = self.calculate_world_to_view_proj_mat(rotation_mat=rotation_mat, translation_vector=translation_vector)
        world_to_view_proj_mat = torch.tensor(world_to_view_proj_mat.transpose(), dtype=torch.float32, device=self.device)

        # calculate camera center in world coordinate
        camera_center = torch.inverse(world_to_view_proj_mat)[3, :3]

        # calculate perspective projection matrices
        perspective_proj_mat = self.calculate_perspective_project_mat(fov_x=fov_x, fov_y=fov_y)
        perspective_proj_mat = torch.tensor(perspective_proj_mat, dtype=torch.float32, device=self.device).transpose(0, 1)

        # calculate world to image projection matrices
        world_to_image_proj_mat = torch.matmul(world_to_view_proj_mat, perspective_proj_mat)

        return {'world_to_view_proj_mat': world_to_view_proj_mat, 'perspective_proj_mat': perspective_proj_mat,
                'world_to_image_proj_mat': world_to_image_proj_mat, 'camera_center': camera_center}

    def calculate_camera_sphere_radius(self) -> float:
        """
        Calculate the radius of the sphere that contains all training camera centers, note that the center of the sphere is the average of all camera centers.
        """
        # calculate all camera centers with shape (3, num_views)
        world_to_view_proj_mats = [self.calculate_world_to_view_proj_mat(rotation_mat=sample.rotation_mat, translation_vector=sample.translation_vec).astype(np.float32) for sample in self.train_samples.values()]
        camera_centers = np.stack([np.linalg.inv(mat)[:3, -1] for mat in world_to_view_proj_mats], axis=1)
        # calculate the average of all camera centers
        average_camera_center = np.mean(camera_centers, axis=1, keepdims=True)  # shape (3, 1)
        # calculate the distance between all camera centers and the average of all camera centers
        distance = np.linalg.norm(camera_centers - average_camera_center, axis=0, keepdims=True)  # shape (1, num_views)
        # calculate the radius of the sphere that contains all camera centers
        radius = np.max(distance).item() * 1.1  # amplify the radius by 1.1
        return radius

    @staticmethod
    def calculate_world_to_view_proj_mat(rotation_mat: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        """
        Calculate world to view projection matrix (actually extrinsic matrix).
        :param rotation_mat: rotation matrix with shape (3, 3).
        :param translation_vector: translation vector with shape (3,).
        :return camera_mat: camera matrix with shape (4, 4).
        """
        world_to_view_proj_mat = np.concatenate([rotation_mat.transpose(), translation_vector.reshape(3, 1)], axis=1)  # shape (3, 4)
        world_to_view_proj_mat = np.concatenate([world_to_view_proj_mat, np.array([[0, 0, 0, 1.]])], axis=0)  # shape (4, 4)
        return world_to_view_proj_mat.astype(np.float32)

    def calculate_perspective_project_mat(self,  fov_x: float, fov_y: float) -> np.ndarray:
        """
        Calculate perspective projection matrix
        :param fov_x: field of view in x direction.
        :param fov_y: field of view in y direction.
        :return: perspective projection matrix with shape (4, 4).
        """
        tan_half_fov_x, tan_half_fov_y = np.tan(fov_x / 2), np.tan(fov_y / 2)
        proj_mat = np.array([
            [1. / tan_half_fov_x, 0, 0, 0], [0, 1. / tan_half_fov_y, 0, 0],
            [0, 0, self.z_far / (self.z_far - self.z_near), -self.z_far * self.z_near / (self.z_far - self.z_near)], [0, 0, 1, 0]])
        return proj_mat

    def read_extrinsic_binary(self, extrinsic_filepath: str) -> dict:
        """
        Read extrinsic parameters from binary file.
        """
        extrinsic_info = {}
        with open(extrinsic_filepath, mode='rb') as fid:
            num_views = self.read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            for _ in range(num_views):
                data = self.read_next_bytes(fid, num_bytes=64, format_char_sequence='idddddddi')
                view_id, rotation_quaternion_vector, translation_vector, camera_id = data[0], np.array(data[1:5]), np.array(data[5:8]), data[8]
                rotation_mat = self.quaternion_to_rotation_matrix(quaternion=rotation_quaternion_vector).transpose()

                img_name = ""
                current_char = self.read_next_bytes(fid, 1, 'c')[0]
                while current_char != b'\x00':  # look for the ASCII 0 entry
                    img_name += current_char.decode('utf-8')
                    current_char = self.read_next_bytes(fid, 1, 'c')[0]

                num_points_2d = self.read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
                _ = self.read_next_bytes(fid, num_bytes=24 * num_points_2d, format_char_sequence='ddq' * num_points_2d)

                extrinsic_info[view_id] = {'view_id': view_id, 'camera_model_id': camera_id, 'img_name': img_name,
                                           'rotation_mat': rotation_mat, 'translation_vector': translation_vector}
        return extrinsic_info

    def read_intrinsic_binary(self, intrinsic_path: str) -> dict:
        """
        Read intrinsic parameters from binary file.
        """
        intrinsic_info = {}
        with open(intrinsic_path, mode='rb') as fid:
            num_camera_models = self.read_next_bytes(fid, num_bytes=8, format_char_sequence='Q')[0]
            for _ in range(num_camera_models):
                data = self.read_next_bytes(fid, num_bytes=24, format_char_sequence='iiQQ')
                _, camera_model_id, width, height = data[0], data[1], data[2], data[3]
                camera_model_name = CAMERA_MODEL_IDS[camera_model_id].model_name
                assert camera_model_name in CAMERA_MODEL_NAMES, f'Camera model {camera_model_name} is not supported.'
                num_camera_model_params = CAMERA_MODEL_IDS[camera_model_id].num_params
                params = self.read_next_bytes(fid, num_bytes=8 * num_camera_model_params, format_char_sequence="d" * num_camera_model_params)
                # calculate field of view
                fov_x = 2 * np.arctan(width / 2 / params[0])
                fov_y = 2 * np.arctan(height / 2 / (params[0] if camera_model_name == 'SIMPLE_PINHOLE' else params[1]))
                intrinsic_info[camera_model_id] = {'fov_x': fov_x, 'fov_y': fov_y, 'width': width, 'height': height}
        return intrinsic_info

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to rotation matrix.
        :param quaternion: quaternion vector with shape (4,).
        :return rotation_mat: rotation matrix with shape (3, 3).
        """
        rotation_mat = np.array([
            [1 - 2 * quaternion[2] ** 2 - 2 * quaternion[3] ** 2, 2 * quaternion[1] * quaternion[2] - 2 * quaternion[0] * quaternion[3], 2 * quaternion[3] * quaternion[1] + 2 * quaternion[0] * quaternion[2]],
            [2 * quaternion[1] * quaternion[2] + 2 * quaternion[0] * quaternion[3], 1 - 2 * quaternion[1] ** 2 - 2 * quaternion[3] ** 2, 2 * quaternion[2] * quaternion[3] - 2 * quaternion[0] * quaternion[1]],
            [2 * quaternion[3] * quaternion[1] - 2 * quaternion[0] * quaternion[2], 2 * quaternion[2] * quaternion[3] + 2 * quaternion[0] * quaternion[1], 1 - 2 * quaternion[1] ** 2 - 2 * quaternion[2] ** 2]])
        return rotation_mat

    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion.
        :param R: rotation matrix with shape (3, 3).
        :return qvec: quaternion vector with shape (4,).
        """
        Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
        K = np.array([
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
        eigvals, eigvecs = np.linalg.eigh(K)
        qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
        if qvec[0] < 0:
            qvec *= -1
        return qvec

    @staticmethod
    def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
        """Read and unpack the next bytes from a binary file.
        :param fid: File object.
        :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
        :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
        :param endian_character: Any of {@, =, <, >, !}
        :return: Tuple of read and unpacked values.
        """
        data = fid.read(num_bytes)
        return struct.unpack(endian_character + format_char_sequence, data)

class AriaDataset(BaseDataset):
    Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
    Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

    def __init__(self, root: str, vrs_path: str, closedloop_path: str, image_folder: str, logger: CustomLogger, device: str, eval_interval: int = 8, z_near: float = 0.01, z_far: float = 100):
        self.vrs_path = os.path.join(root, vrs_path)
        self.closedloop_path = os.path.join(root, closedloop_path)
        self.image_folder = os.path.join(root, image_folder)
        # Call BaseDataset constructor with dummy paths (we override load_samples)
        super().__init__(root="", image_folder="", logger=logger, device=device, eval_interval=eval_interval, z_near=z_near, z_far=z_far)

        semidense_path = os.path.join(root, 'mps', 'slam', 'semidense_points.csv.gz')
        if os.path.exists(semidense_path):
            points, rgbs = self.get_semidense_points(semidense_path)
            self.point_cloud = (np.array(points), np.array(rgbs))
        else:
            self.logger.warning(f"Semidense point cloud not found at {semidense_path}")
            self.point_cloud = None

    def load_samples(self, cam_mat_folder: str, img_folder: str):
        print("AriaDataset: Loading intrinsics...")
        cam_intrinsics = self.get_camera_intrinsics(self.vrs_path)
        print("AriaDataset: Loading extrinsics...")
        cam_extrinsics = self.get_camera_extrinsics(self.vrs_path, self.closedloop_path)
        print(f"AriaDataset: Found {len(cam_extrinsics)} frames. Loading images...")
        print(f"AriaDataset: Image folder: {self.image_folder}")
        samples = []
        for idx, key in enumerate(cam_extrinsics):
            if idx % 100 == 0: print(f"AriaDataset: Loading sample {idx}/{len(cam_extrinsics)}")
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width
            rotation_mat = self.quaternion_to_rotation_matrix(extr.qvec).transpose()
            translation_vec = np.array(extr.tvec)
            focal_length_x = intr.params[0]
            fov_x = 2 * np.arctan(width / 2 / focal_length_x)
            fov_y = 2 * np.arctan(height / 2 / focal_length_x)
            tan_half_fov_x = np.tan(fov_x / 2)
            tan_half_fov_y = np.tan(fov_y / 2)
            img_name = extr.name
            img_path = os.path.join(self.image_folder, img_name)
            if not os.path.exists(img_path):
                if idx == 0: print(f"AriaDataset: First image not found: {img_path}")
                self.logger.warning(f"Image not found: {img_path}")
                continue
            
            world_to_view_proj_mat = self.calculate_world_to_view_proj_mat(rotation_mat, translation_vec)
            camera_center = np.linalg.inv(world_to_view_proj_mat)[:3, -1]
            camera_center = torch.tensor(camera_center, dtype=torch.float32, device=self.device)
            perspective_proj_mat = self.calculate_perspective_project_mat(fov_x, fov_y)
            world_to_view_proj_mat_torch = torch.tensor(world_to_view_proj_mat.transpose(), dtype=torch.float32, device=self.device)
            perspective_proj_mat_torch = torch.tensor(perspective_proj_mat, dtype=torch.float32, device=self.device).transpose(0, 1)
            world_to_image_proj_mat = torch.matmul(world_to_view_proj_mat_torch, perspective_proj_mat_torch)
            sample = DatasetSample(
                cam_idx=idx,
                camera_center=camera_center,
                img_name=img_name,
                img=None,
                img_height=height,
                img_width=width,
                rotation_mat=rotation_mat,
                translation_vec=translation_vec,
                fov_x=fov_x,
                fov_y=fov_y,
                tan_half_fov_x=tan_half_fov_x,
                tan_half_fov_y=tan_half_fov_y,
                world_to_view_proj_mat=world_to_view_proj_mat_torch,
                world_to_image_proj_mat=world_to_image_proj_mat,
                perspective_proj_mat=perspective_proj_mat_torch
            )
            samples.append(sample)
        print("AriaDataset: Finished loading samples.")
        return samples

    def get_camera_intrinsics(self, vrs_path):
        if vrs_path is None :
            # ASE data set case: use calibration from get_ase_rgb_calibration
            rgb_calib = get_ase_rgb_calibration()
            width, height = rgb_calib.get_image_size()
            focal_length = rgb_calib.get_focal_lengths()[0]
            principal_point = rgb_calib.get_principal_point()
            cameras = {
                0: self.Camera(
                    id=0,
                    model="SIMPLE_PINHOLE",
                    width=int(width),
                    height=int(height),
                    params=np.array([focal_length, *principal_point])
                )
            }
        else :
            # ADT data set case: Extract intrinsics from VRS file
            provider = data_provider.create_vrs_data_provider(vrs_path)
            rgb_stream_id = provider.get_stream_id_from_label('camera-rgb')

            # Get the device calibration, if it exists
            device_calib = provider.get_device_calibration()
            rgb_calib = device_calib.get_camera_calib("camera-rgb")
            rgb_projection_params = rgb_calib.projection_params()
            rgb_focal_length = rgb_calib.get_focal_lengths()[0]
            rgb_principal_point = rgb_calib.get_principal_point()

            cameras = {
                0: self.Camera(
                    id=0,
                    model="SIMPLE_PINHOLE",
                    width=int(rgb_calib.get_image_size()[0]),
                    height=int(rgb_calib.get_image_size()[1]),
                    params=np.array([rgb_focal_length, *rgb_principal_point])
                )
            }
        return cameras

    def get_camera_extrinsics(self, vrs_path, closedloop_path):
        if vrs_path is None :
            # ASE data set case: T_rgb_device is hardcoded
            rgb_calib = get_ase_rgb_calibration()
            T_device_rgb = rgb_calib.get_transform_device_camera().to_matrix()
            T_rgb_device = np.linalg.inv(T_device_rgb)

            # From ase_tutorial_notebook.ipynb
            def _transform_from_Rt(R, t):
                M = np.identity(4)
                M[:3, :3] = R
                M[:3, 3] = t
                return M

            # From ase_tutorial_notebook.ipynb
            def _read_trajectory_line(line):
                line = line.rstrip().split(",")
                pose = {}
                pose["timestamp"] = int(line[1])
                translation = np.array([float(p) for p in line[3:6]])
                quat_xyzw = np.array([float(o) for o in line[6:10]])
                rot_matrix = self.quaternion_to_rotation_matrix(quat_xyzw[[3, 0, 1, 2]])  # Convert from xyzw to wxyz
                rot_matrix = np.array(rot_matrix)
                pose["position"] = translation
                pose["rotation"] = rot_matrix
                pose["transform"] = _transform_from_Rt(rot_matrix, translation)

                return pose

            # From ase_tutorial_notebook.ipynb
            def read_trajectory_file(filepath):
                assert os.path.exists(filepath), f"Could not find trajectory file: {filepath}"
                with open(filepath, 'r') as f:
                    _ = f.readline() # Header
                    positions = []
                    rotations = []
                    transforms = []
                    timestamps = []
                    for line in f.readlines():
                        pose = _read_trajectory_line(line)
                        positions.append(pose["position"])
                        rotations.append(pose["rotation"])
                        transforms.append(pose["transform"])
                        timestamps.append(pose["timestamp"])
                    positions = np.stack(positions)
                    rotations = np.stack(rotations)
                    transforms = np.stack(transforms)
                    timestamps = np.array(timestamps)
                print(f"Loaded trajectory with {len(timestamps)} device poses.")
                return {
                    "ts": positions,
                    "Rs": rotations,
                    "Ts_world_from_device": transforms,
                    "timestamps": timestamps,
                }

            # ASE claims to use the same MPS format as ADT, but it doesn't, so I have to do stupid stuff like this
            closed_loop_traj = read_trajectory_file(closedloop_path)

            images = {}

            # ASE data set case: each trajectory pose corresponds directly to an RGB frame
            for i, T_world_device in enumerate(closed_loop_traj["Ts_world_from_device"]):
                T_device_world = np.linalg.inv(T_world_device)
                T_rgb_world = T_rgb_device @ T_device_world
                # T_world_rgb = np.linalg.inv(T_rgb_world)
                T_world_rgb = T_rgb_world # IDK why this is like this, but it works experimentally

                # Extract the rotation quaternion and translation components of the transformation matrix
                R = T_world_rgb[:3, :3]
                T = T_world_rgb[:3, 3]
                qvec = self.rotation_matrix_to_quaternion(R)

                images[i] = self.Image(
                    id=i,
                    qvec=qvec,
                    tvec=T,
                    camera_id=0,
                    name=f"vignette{i:07d}.png",
                    xys=None,
                    point3D_ids=None
                )
        else :
            # ADT data set case: Extract extrinsics from VRS file
            provider = data_provider.create_vrs_data_provider(vrs_path)
            rgb_stream_id = provider.get_stream_id_from_label('camera-rgb')

            # Get the device calibration, if it exists
            device_calib = provider.get_device_calibration()
            rgb_calib = device_calib.get_camera_calib("camera-rgb")
            T_device_rgb = rgb_calib.get_transform_device_camera().to_matrix()
            T_rgb_device = np.linalg.inv(T_device_rgb)

            # Read the MPS closed loop trajectory
            closed_loop_traj = read_closed_loop_trajectory(closedloop_path)

            # ADT data set case: each RGB frame timestamp needs to be matched to the nearest trajectory pose
            traj_timestamps = np.array([p.tracking_timestamp.total_seconds() * 1e9 for p in closed_loop_traj])

            def get_nearest_pose_idx(timestamps, query_timestamp_ns):
                idx = np.searchsorted(timestamps, query_timestamp_ns)
                if idx == 0:
                    return 0
                if idx == len(timestamps):
                    return len(timestamps) - 1
                
                # Check which one is closer: idx or idx-1
                if abs(timestamps[idx] - query_timestamp_ns) < abs(timestamps[idx-1] - query_timestamp_ns):
                    return idx
                else:
                    return idx - 1

            images = {}

            for i in range(0, provider.get_num_data(rgb_stream_id)):
                sensor_data = provider.get_sensor_data_by_index(rgb_stream_id, i)
                sensor_time_ns = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)

                # Get the nearest pose from the closed loop trajectory
                nearest_idx = get_nearest_pose_idx(traj_timestamps, sensor_time_ns)
                nearest_pose = closed_loop_traj[nearest_idx]
                T_world_device = nearest_pose.transform_world_device.to_matrix()
                T_device_world = np.linalg.inv(T_world_device)
                T_rgb_world = T_rgb_device @ T_device_world
                # T_world_rgb = np.linalg.inv(T_rgb_world)
                T_world_rgb = T_rgb_world # IDK why this is like this, but it works experimentally

                # Extract the rotation quaternion and translation components of the transformation matrix
                R = T_world_rgb[:3, :3]
                T = T_world_rgb[:3, 3]
                qvec = self.rotation_matrix_to_quaternion(R)

                images[i] = self.Image(
                    id=i,
                    qvec=qvec,
                    tvec=T,
                    camera_id=0,
                    name=f"{i:07d}.png",
                    xys=None,
                    point3D_ids=None
                )
        
        return images

    def get_semidense_points(self, semidense_path):
        points = read_global_point_cloud(semidense_path, StreamCompressionMode.GZIP)
        points = [p.position_world.tolist() for p in points]

        # Colors aren't provided with Aria reconstructions, so just pass (128, 128, 128) for all points
        rgbs = [[128, 128, 128] for _ in points]

        return points, rgbs