import os
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn import metrics 

from Models.base_model import base_model

class MLP(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--embed_size",type=int,default=100,help="Hidden layer size for MLP.")
        parser.add_argument("--hidden_size",type=int,default=100,help="Hidden layer size for MLP.")
        parser.add_argument("--epoches",type=int,default=3000,help="Max iteractions")
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        hidden_size = [args.embed_size]
        if args.hidden_size>1:
            hidden_size.append(args.hidden_size)
        
        self.classifier = MLPClassifier(hidden_layer_sizes=hidden_size,learning_rate_init=args.lr,
            max_iter=args.epoches, learning_rate="adaptive", random_state=args.random_seed, 
            verbose=False, solver="adam",alpha=5e-4)
    
    def model_predict(self, X):
        self.classifier.predict(X)
