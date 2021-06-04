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


# class LR(base_model):
#     @staticmethod
#     def parse_model_args(parser):
#         parser.add_argument("--epoches",type=int,default=1000,help="Max iteractions")
#         parser.add_argument("--tol",type=float,default=1e-4,help="Max toleration.")
#         parser.add_argument("--solver",type=str,default="lbfgs")
#         return base_model.parse_model_args(parser)

#     def __init__(self, args, feature_dim=20):
#         self.linear = torch.nn.Linear(feature_dim, 1)
#         self.sigmoid = torch.nn.Sigmoid()
#         self.feature_type = args.feature_file
#         self.model_path = "Checkpoints/LR/"
#         self.lr = args.lr
#         self.epoch = args.epoches
    
#     def forward(self, x):
#         y_pred = self.sigmoid(self.linear(x))
    
#     def model_predict(self,x):
#         return self.forward(x)
    
#     def fit(self, X, y, model):
#         criterion = torch.nn.BCELoss(size_average=True)
#         optimizer = torch.optim.Adam(model.parameters(),lr=self.lr)
#         for epoch in range(self.epoch):
#             y_pred = self.forward(X)
#             loss = criterion(y_pred, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     def predict(self, X, y, print_recall=False, no_pred=False):
#         pred = self.model_predict(X) >0.5
#         # Accuracy
#         acc = metrics.accuracy_score(y, pred)
#         # AUC
#         auc = metrics.roc_auc_score(y,prob)
#         # F1 score for churn users
#         f1 = metrics.f1_score(y,pred)
#         # Precision and Recall
#         precision = metrics.precision_score(y,pred)
#         recall = metrics.recall_score(y, pred)
#         if print_recall:
#             print("acc: {0:.3f}, auc: {1:.3f} f1: {2:.3f}".format(acc,auc,f1))
#             print(metrics.classification_report(y,pred,digits=3))
#         return [auc,acc,f1, precision, recall]


#     def save_model(self, model_path=None):
#         if model_path is None:
#             model_path = self.model_path
#         utils.check_dir(os.path.join(model_path,"LR.pkl"))
#         logging.info('Save model to ' + model_path[:50] + '...')     

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