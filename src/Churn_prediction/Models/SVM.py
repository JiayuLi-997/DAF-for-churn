import os
import logging
import argparse
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import metrics 

from Models.base_model import base_model

class SVM(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--C",type=float,default=1.0)
        parser.add_argument("--kernel",type=str,default='rbf',
            help="Kernel for SVM classifier.")
        parser.add_argument("--gamma", type=float,default=-1,
            help="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.")
        parser.add_argument("--degree",type=int,default=3,
            help="Degree of the polynomial kernel function (‘poly’).")
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        if args.kernel=='linear':
            self.classifier = svm.SVC(C=args.C, kernel = args.kernel)
        else:
            gamma = args.gamma
            if gamma == -1:
                gamma='scale'
            if args.kernel in ["rbf","sigmoid"]:
                self.classifier = svm.SVC(C=args.C, kernel = args.kernel,gamma=gamma)
            if args.kernel =="poly":
                self.classifier = svm.SVC(C=args.C, kernel = args.kernel,gamma=gamma,degree=args.degree)
    
    def model_predict(self, X):
        self.classifier.predict(X)
