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
import torch.autograd as autograd
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--word_dim', dest='word_dim', type=int, help='word embedding dimension', default=300)
parser.add_argument('--hid_size', dest='hid_size', type=int, help='hidden dimension size', default=500)
parser.add_argument('--enc_size', dest='enc_size', type=int, help='encode dimension size', default=10)
parser.add_argument('--epochs', dest='epochs', type=int, help='epochs for model training', default=2000)
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
        
        self.WeC2P = nn.Linear(2 * word_dim, word_dim)
        self.WeP2H = nn.Linear(word_dim, hid_size)

        self.WeH2M = nn.Linear(hid_size, enc_size)
        self.WeH2D = nn.Linear(hid_size, enc_size)
        
        self.WdE2H = nn.Linear(enc_size, hid_size)
        self.WdH2P = nn.Linear(hid_size, word_dim)
        self.WdP2C = nn.Linear(word_dim, 2*word_dim)
        
        self.tanh = nn.Tanh()
        
    def encode(self, sent, node, size):
        parent = Variable(torch.FloatTensor(torch.randn(1, size)))
        
        for i in range(node):
            if i == 0:
                parent = sent[i]
            else: 
                parent = self.tanh(self.WeC2P(torch.cat((parent, sent[i]), 0)))
        hid_code = self.tanh(self.WeP2H(parent))
        mu = self.tanh(self.WeH2M(hid_code))
        va = self.tanh(self.WeH2D(hid_code))

        return hid_code, mu, va
    
    def reparameterize(self, mu, logvar):
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          
          return eps.mul(std).add_(mu)
    
    def decode(self, code, node, size):
        sent = Variable(torch.FloatTensor(torch.randn(node, size)).zero_())
        
        parent = self.tanh(self.WdH2P(self.tanh(self.WdE2H(code))))
        
        for i in range(node-1): 
            if i < node - 2:           
                children = self.tanh(self.WdP2C(parent))
                sent[node-i-1] = children[0:size]
                parent = children[size:]
            else:
                children = self.tanh(self.WdP2C(parent))
                sent[1] = children[0:size]
                sent[0] = children[size:]
        
        return sent
    
    def forward(self, sent):
        node = sent.size()[0]
        size = sent.size()[1]
        hi, mu, va = self.encode(sent, node, size)
        zc = self.reparameterize(mu, va)
        se = self.decode(zc, node, size)
        return se
    
rvae = RVAE(args.word_dim, args.hid_size, args.enc_size)

mse_criterion = nn.MSELoss()
optimizer = optim.Adam(rvae.parameters(), lr=args.learning_rate)

def main():
    sent = Variable(torch.FloatTensor(torch.randn(50, args.word_dim)))
    loss_storage = []
    
    for i in range(args.epochs):
        out = rvae(sent)
        loss = mse_criterion(out, sent)
        loss_storage.append(loss.data[0])
        print(loss)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        if i == args.epochs - 1:
            print(sent)
            print(out)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            output = cos(sent, out)
            print(output)
    
    plt.plot(range(args.epochs), loss_storage, label="loss", color="blue")
    plt.legend()
    plt.show() 
    
if __name__ == "__main__":
    main()
