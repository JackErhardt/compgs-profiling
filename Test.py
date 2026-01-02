import argparse
import random

import numpy as np
import torch

from Modules import TesterCompGS
from Modules.Common import BaseDataset, AriaDataset, CustomLogger, init


# fix random seed
def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    setup_seed(seed=3407)

    parser = argparse.ArgumentParser('\nEval models, if you want to override configurations, please use --key=value')
    parser.add_argument('--config', type=str, help='filepath of configuration files', required=True)
    parser.add_argument('--experiment_root', type=str, help='experiment root path', required=True)
    parser.add_argument('--device', type=str, help='device to use', default='cuda')
    parser.add_argument('--load_dir', type=str, help='directory containing bitstreams.npz and weights.pth', default=None)
    parser.add_argument('--save_dir', type=str, help='directory to save rendered images and results', default=None)

    args, override_cfgs = parser.parse_known_args()
    override_cfgs = dict(arg.lstrip('-').split('=') for arg in override_cfgs) if len(override_cfgs) > 0 else {}

    # Note: The init function creates a new logger and experiment directory, which we don't need for testing.
    # We'll use the provided experiment_root and a dummy logger.
    configs, _, _ = init(config_path=args.config, override_cfgs=override_cfgs)

    dummy_logger = CustomLogger(experiment_dir='Dummy', enable_detail_log=False)
    
    dataset_configs = configs['dataset']
    if dataset_configs['sfm_type'] == 'aria':
        dataset = AriaDataset(
            root=dataset_configs['root'],
            vrs_path=dataset_configs['vrs_path'],
            closedloop_path=dataset_configs['closedloop_path'],
            image_folder=dataset_configs['image_folder'],
            logger=dummy_logger,
            device=args.device,
            eval_interval=dataset_configs['eval_interval']
        )
    else:
        dataset = BaseDataset(
            root=dataset_configs['root'], 
            image_folder=dataset_configs['image_folder'], 
            logger=dummy_logger, 
            device=args.device,
            eval_interval=dataset_configs['eval_interval']
        )

    tester_args = {
        'device': args.device, 
        'experiment_root': args.experiment_root, 
        'dataset': dataset,
        'gpcc_codec_path': configs['training']['gpcc_codec_path'],
        'eval_lpips': configs['training']['eval_lpips'],
        'load_dir': args.load_dir,
        'save_dir': args.save_dir
    }
    
    # Add gaussian model parameters to tester_args
    for key, value in configs['gaussians'].items():
        tester_args[key] = value
    
    tester = TesterCompGS(**tester_args)
    tester.eval()
