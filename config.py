# arguments configuration
import argparse

parser = argparse.ArgumentParser(description='This is our final project')

# hyperparmeters
parser.add_argument('--max_epoch', type=int, default=50, help='max epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of training data')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--decay', type=float, default=1e-5, help='weight decay')

# file directory
parser.add_argument('--datapath', type=str, default='./dataset', help='relative path to dataset folder')
parser.add_argument('--model_save', type=str, default='./test1', help='folder to save best model\'s .pth file')
parser.add_argument('--log_save', type=str, default='./test1', help='folder to save csvlog file')


# arguments variable
args = parser.parse_args()