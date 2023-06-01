# hyperparameter configurations
import argparse

def args():
    parser = argparse.ArgumentParser(description='This is our final project')
    parser.add_argument('--epoch', )
    parser.add_argument('--lr', )
    parser.add_argument('--decay', )
    parser.add_argument('--batch_size', )


    return parser.parse_args()