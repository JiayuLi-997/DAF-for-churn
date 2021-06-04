import os
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics 

from Models.GBDT import GBDT
from helpers import utils
import joblib

class RF(GBDT):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--max_features",type=float, default=1.0,
                            help="The number of features considered in each time of splitting.")
        return GBDT.parse_model_args(parser)

    def __init__(self, args):
        self.classifier = RandomForestClassifier(random_state=args.random_seed, oob_score=True,
            n_estimators=args.estimators, max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf, max_features=args.max_features)
        self.model_path="Checkpoints/RF/"
        self.feature_type = args.feature_file
    
    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path,"RF.pkl"))
        joblib.dump(self.classifier, os.path.join(model_path,"RF_%s.pkl"%(self.feature_type)))
        logging.info('Save model to ' + model_path[:50] + '...')     