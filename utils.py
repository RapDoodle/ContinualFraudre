import scipy.sparse as sp
import argparse
import logging
import numpy as np
import random
import logging

import torch

def parse_argument():
    parser = argparse.ArgumentParser(description = 'pytorch version of GraphSAGE')
    # data options
    parser.add_argument('--data', type = str, default = 'cora')
    
    parser.add_argument('--num-epochs', type = int, default = 100)
    parser.add_argument('--batch-size', type = int, default = 128)
    parser.add_argument('--seed', type = int, default = 13)
    parser.add_argument('--cuda', action = 'store_true', help = 'use CUDA')
    parser.add_argument('--num-neg-samples', type = int, default = 10)
    parser.add_argument('--num-layers', type = int, default = 2)
    parser.add_argument('--embed-size', type = int, default = 64)
    parser.add_argument('--learning-rate', type = float, default = 0.1)
    parser.add_argument('--normalize', action = 'store_true', default = False)

    parser.add_argument('--detect-strategy', type = str, default = 'bfs')  # 'simple' / 'bfs'
    parser.add_argument('--new-ratio', type = float, default = 0.0)

    parser.add_argument('--memory-size', type = int, default = 0)
    parser.add_argument('--memory-strategy', type = str, default = 'class')   # 'random' / 'class'
    parser.add_argument('--p', type = float, default = 1)
    parser.add_argument('--alpha', type = float, default = 0.0)
    
    parser.add_argument('--ewc-lambda', type = float, default = 0.0)
    parser.add_argument('--ewc-type', type = str, default = 'ewc')  # 'l2' / 'ewc'

    parser.add_argument('--eval', action = 'store_true')

    parser.add_argument('--max-detect-size', type = int, default = None)

    # Arguments from FRAUDRE
    parser.add_argument('--lambda-1', type=float, default=1e-4, help='Weight decay (L2 loss weight).')
    parser.add_argument('--embed-dim', type=int, default=64, help='Node embedding size at the first layer.')
    parser.add_argument('--test-epochs', type=int, default=10, help='Epoch interval to run test set.')
    parser.add_argument('--skip-ewc', action = 'store_true', default = False)

    args = parser.parse_args()

    return args

def print_args(args):
    config_str = 'Parameters: '
    for name, value in vars(args).items():
        config_str += str(name) + ': ' + str(value) + '; '
    logging.info(config_str)


def check_device(cuda):
    if torch.cuda.is_available():
        if not cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            device_id = torch.cuda.current_device()
            print('using device', device_id, torch.cuda.get_device_name(device_id))
    device = torch.device("cuda" if cuda else "cpu")
    logging.info('Device:' + str(device))
    return device


def node_classification(trut, pred, name = ''):
    from sklearn import metrics
    f1 = np.round(metrics.f1_score(trut, pred, average="macro"), 6)
    acc = np.round(metrics.f1_score(trut, pred, average="micro"), 6)
    logging.info(name + '   Macro F1:' +  str(f1) \
            + ";    Micro F1:" +  str(acc))
    return f1, acc


def normalize(mx):

	"""Row-normalize sparse matrix"""

	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def read_stream_statistics(path):
    with open(path, 'r') as f:
        stats = f.read().split('\n')
    stream_stats = {}
    for stat in stats:
        splitted = stat.split('=')
        if len(splitted) == 2:
            key = splitted[0]
            val = splitted[1]
            # Convert to int when possible
            try:
                val = int(val)
            except:
                pass
            stream_stats[key] = val
    return stream_stats