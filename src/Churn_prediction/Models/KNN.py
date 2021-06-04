import os
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics 

from Models.base_model import base_model
from helpers import utils
import joblib

class KNN(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--n_neighbors",type=int,default=5)
        parser.add_argument("--leaf_size",type=int,default=30)
        parser.add_argument("--algorithm",type=str,default='auto',help="Algorithm used to compute the nearest neighbors.")
    
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        self.classifier = KNeighborsClassifier(n_neighbors=args.n_neighbors,
            leaf_size=args.leaf_size, algorithm=args.algorithm)
        self.model_path = "Checkpoints/KNN/"
        self.feature_type = args.feature_file

    def model_predict(self, X):
        self.classifier.predict(X)

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path,"KNN.pkl"))
        joblib.dump(self.classifier, os.path.join(model_path,"KNN_%s.pkl"%(self.feature_type)))
        logging.info('Save model to ' + model_path[:50] + '...')     