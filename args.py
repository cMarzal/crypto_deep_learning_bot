from __future__ import absolute_import
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='binance autobot arguments')

    # training parameters
    parser.add_argument('--gpu', default=0, type=int,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of validation iterations")
    parser.add_argument('--val_epoch', default=1, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=500, type=int,
                    help="test batch size")
    parser.add_argument('--lr', default=0.000001, type=float,
                    help="initial learning rate")
    parser.add_argument('--beta', default=0.5, type=float,
                        help="initial beta")
    parser.add_argument('--weight-decay', default=0.0005, type=float,
                    help="initial learning rate")


    # others
    parser.add_argument('--prediction', type=int, default=48)
    parser.add_argument('--period', type=int, default=2880)
    parser.add_argument('--step', type=int, default=12)
    parser.add_argument('--percentage_buy', type=int, default=2)

    args = parser.parse_args()

    return args
