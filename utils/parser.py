import argparse
import os

parser = argparse.ArgumentParser(description='SPPNet PyTorch Training')


# ========================== Indispensable arguments ==========================

# ============================= Optical arguments =============================

# Working mode arguments
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-f', '--finetune', dest='finetune', action='store_true', 
                    help='fine tune the model')
parser.add_argument('--pretrained', type=str, default=None,
                    help='using locally pre-trained model. The path of pre-trained model should be given')
parser.add_argument('--resume', type=str, metavar='PATH', default=None,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--seed-data', default=2026, type=int,
                    help='seed for initializing dataset. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--cpu', action='store_true',
                    help='disable GPU training (default: False)')
parser.add_argument('--cpu-affinity', default=None, type=str,
                    help='CPU affinity, like "0xffff"')

# Other arguments
parser.add_argument('--epochs', type=int, metavar='N', default=150,
                    help='number of total epochs to run')
parser.add_argument('-a', '--alpha', type=float, default=0,
                    help='alpha of loss function')
parser.add_argument('--mask-MSE', dest='mask_MSE', action='store_true', default=False,
                    help='use masked MSE as loss function (default: False)')
parser.add_argument('--mask-IG', dest='mask_IG', action='store_true', default=False,
                    help='use masked image gradient loss as loss function (default: False)')
parser.add_argument('--MSE-mask-range', type=float, nargs=2, default=None,
                    help='range for masked MSE (default: None)')
parser.add_argument('--MSE-mask-weight', type=float, default=None,
                    help='weight for masked MSE (default: None)')
parser.add_argument('--IG-mask-range', type=float, nargs=2, default=None,
                    help='range for masked IG loss (default: None)')
parser.add_argument('--IG-mask-weight', type=float, default=None,
                    help='weight for masked IG loss (default: None)')
parser.add_argument('--scheduler', type=str, default='cosine', choices=['const', 'cosine'],
                    help='learning rate scheduler')
parser.add_argument('-b', '--batch-size', type=int, default=20, metavar='N',
                    help='mini-batch size')
parser.add_argument('-j', '--workers', type=int, default=0, metavar='N',
                    help='number of data loading workers')
parser.add_argument('-n', '--train-val-num', type=int, default=125, metavar='N',
                    help='number of train and valid data')
parser.add_argument('--test-num', type=int, default=500, metavar='N',
                    help='number of test data')
parser.add_argument('-s', '--scenario', type=str, default='munk_A',
                    help='munk profile scenario of the dataset')
parser.add_argument('--range', type=int, default=100,
                    help='range of the TL area')
parser.add_argument('--finetune-scenario', type=str, default='munk_B',
                    help='munk profile scenario of the dataset for fine-tuning')
parser.add_argument('--finetune-range', type=int, default=100,  
                    help='range of the TL area for fine-tuning')
parser.add_argument('--bty', type=str, default='flat', 
                    help='bottom type of the TL area')
parser.add_argument('--snr-db', type=float, default=0,
                    help='SNR value in dB for adding noise to the data')
# def get_root_dir():
#     return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

parser.add_argument('--root-dir', type=str, default='/home/zhizhen/UWC/SPPNet_finetune_free_nr_ig_scale_mask/',
                    help='the path of root directory')

args = parser.parse_args()