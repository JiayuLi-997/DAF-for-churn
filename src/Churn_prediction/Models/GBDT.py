import os
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics 

from Models.base_model import base_model
from helpers import utils
import joblib

class GBDT(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--estimators", type=int,default=200,
                            help="The number of boosting stages to perform.")
        parser.add_argument("--subsample",type=float,default=1,
                            help="The fraction of samples to be used for fitting the individual base learners.(<=1.0)")
        parser.add_argument("--max_depth", type=int,default=3,
                            help="Maximum depth of the individual regression estimators.")
        parser.add_argument("--min_samples_split", type=int,default=2,
                            help="The minimum number of samples required to split an internal node.")
        parser.add_argument("--min_samples_leaf", type=int,default=1,
                            help="The minimum number of samples required to be at a leaf node.")
        # parser.add_argument("--max_features",type=int,default=7,
        #                     help="The number of features to consider when looking for the best split.")
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        self.classifier = GradientBoostingClassifier(random_state=args.random_seed,
                               learning_rate=args.lr, n_estimators=args.estimators,
                               max_depth=args.max_depth, min_samples_split=args.min_samples_split,
                               min_samples_leaf=args.min_samples_leaf,subsample=args.subsample)
        self.model_path = "Checkpoints/GBDT/"
        self.feature_type = args.feature_file

    def model_predict(self, X):
        self.classifier.predict(X)

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path,"GBDT.pkl"))
        joblib.dump(self.classifier, os.path.join(model_path,"GBDT_%s.pkl"%(self.feature_type)))
        logging.info('Save model to ' + model_path[:50] + '...')     