import os
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics 

from Models.base_model import base_model
from helpers import utils
import joblib

import torch
from sklearn import metrics


class LR(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--epoches",type=int,default=1000,help="Max iteractions")
        parser.add_argument("--tol",type=float,default=1e-4,help="Max toleration.")
        parser.add_argument("--solver",type=str,default="lbfgs")
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        self.classifier = LogisticRegression(max_iter=args.epoches,tol=args.tol, solver=args.solver,
                                            random_state=args.random_seed)
        self.feature_type = args.feature_file
        self.model_path = "Checkpoints/LR/"
    
    def model_predict(self, X):
        self.classifier.predict(X)
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path,"LR.pkl"))
        joblib.dump(self.classifier, os.path.join(model_path,"LR_%s.pkl"%(self.feature_type)))
        logging.info('Save model to ' + model_path[:50] + '...')     