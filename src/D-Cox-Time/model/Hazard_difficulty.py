import numpy as np
import pandas as pd

import torch
import torchtuples as tt
from torch import nn
import torch.nn.functional as F

from tqdm.notebook import tqdm

class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input):
        return torch.sign(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1,1)

class Hazard_net(nn.Module):
    def __init__(self, feature_num, length):
        super().__init__()
        beta = torch.randn((feature_num, length)).float()
        self.beta = nn.Parameter(beta)
        
        self.feature_num = feature_num
        self.length = length
        
    def forward(self, x_input , t):
        # x_input: batch_size * feature_num * length
        x_input = x_input.float()
        y = self.time_mul(x_input,t)  # y: batch_size * 1
        return y 

    def time_mul(self, x_input, t):
        # x_input: batch_size * feature * length
        # t: batch_size * 1
        x_beta = (x_input*self.beta).sum(dim=1) # batch_size*length
        y = torch.gather(x_beta,dim=1,index=t.long()-1)
        return y
    
    def predict(self, x_input, t):
        return self.forward(x_input, t)
    