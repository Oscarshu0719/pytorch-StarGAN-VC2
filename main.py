import argparse
import os
from torch.backends import cudnn

from data_loader import data_loader
from solver import Solver

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    data_loader_ = data_loader(config.data_dir, batch_size=config.batch_size, mode=config.mode, num_workers=config.num_workers)

    solver = Solver(data_loader_, config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'convert':
        solver.convert()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    dataset_default = 'VCC2016'

    parser.add_argument('--num_spk', type=float, default=4, help='Number of speakers.')
    parser.add_argument('--dataset', type=str, default=dataset_default, choices=['VCC2016', 'VCC2018'], 
        help='Available datasets: VCC2016 and VCC2018 (Default: VCC2016).')
    
    parser.add_argument('--lambda_cyc', type=float, default=10, help='Weight of cycle loss.')
    parser.add_argument('--lambda_gp', type=float, default=5, help='Weight of gradient penalty.')
    parser.add_argument('--lambda_id', type=float, default=5, help='Weight of identity loss.')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Mini-batch size.')
    parser.add_argument('--num_iters', type=int, default=100000, help='Number of total iterations for training discriminator.')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='Number of iterations for decaying learning rate.')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='Learning rate of generator.')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='Learning rate of discriminator.')
    parser.add_argument('--n_critic', type=int, default=5, help='Number of epochs discriminator update for one generator update.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer.')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for Adam optimizer.')
    parser.add_argument('--resume_iters', type=int, default=None, help='The start step of training.')
    
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000 , help='The start step of testing.')
    parser.add_argument('--src_speaker', type=str, default="TM1", help='Source speaker for testing.')
    parser.add_argument('--trg_speaker', type=str, default="['TM1', 'SF1']", help='String list representation of target speakers eg. "[a,b]".')

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'convert'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--test_dir', type=str, default='data/spk_test')
    parser.add_argument('--log_dir', type=str, default='outputs/logs')
    parser.add_argument('--model_save_dir', type=str, default='outputs/models')
    parser.add_argument('--sample_dir', type=str, default='outputs/samples')
    parser.add_argument('--result_dir', type=str, default='outputs/results')
    
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=2000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=100000)

    config = parser.parse_args()
    print(config)
    main(config)
    
