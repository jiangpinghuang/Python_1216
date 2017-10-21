import os
import math
import time
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from torch.autograd import variable
from email.policy import default

parser = argparse.ArgumentParser()

parser.add_argument('--word_dim', dest='word_dim', type=int, help='word embedding dimension', default=20)
parser.add_argument('--hid_size', dest='hid_size', type=int, help='hidden dimension size', default=50)
parser.add_argument('--enc_size', dest='enc_size', type=int, help='encode dimension size', default=20)
parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=10)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning_rate', default=1e-3)
parser.add_argument('--init_weight', dest='init_weight', type=float, help='initial weight for OOV', default=1e-2)
parser.add_argument('--seed', dest='seed', type=int, help='random seed', default=2718281828)
parser.add_argument('--emb_file', dest='emb_file', type=str, help='embedding file', default='')
parser.add_argument('--data_file', dest='data_file', type=str, help='data file', default='')

args = parser.parse_args()
use_gpu = torch.cuda.is_available()

print(args)

torch.manual_seed(args.seed)

class RVAE(nn.Module):
    def __init__(self, word_dim, hid_size, enc_size):
        super(RVAE, self).__init__()


