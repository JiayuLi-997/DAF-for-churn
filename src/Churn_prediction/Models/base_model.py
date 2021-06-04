import numpy as np
import pandas as pd

from sklearn import metrics
import os
import logging

class base_model:
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--lr", type=float, default=0.02,
                            help="Learning rate")
        return parser

    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y):
        self.classifier.fit(X, y)
    
    def predict(self, X, y, print_recall=False, no_pred = False):
        pred = self.classifier.predict(X)
        try:
            prob = self.classifier.predict_proba(X)[:,1]
        except:
            prob = pred
        if no_pred:
            pred = pred>0.5
        # Accuracy
        acc = metrics.accuracy_score(y, pred)
        # AUC
        auc = metrics.roc_auc_score(y,prob)
        # F1 score for churn users
        f1 = metrics.f1_score(y,pred)
        # Precision and Recall
        precision = metrics.precision_score(y,pred)
        recall = metrics.recall_score(y, pred)
        if print_recall:
            print("acc: {0:.3f}, auc: {1:.3f} f1: {2:.3f}".format(acc,auc,f1))
            print(metrics.classification_report(y,pred,digits=3))
        return [auc,acc,f1, precision, recall]

    def test(self, data_loader, no_pred=False):
        train_r = self.predict(data_loader.X_train, data_loader.y_train,no_pred=no_pred)
        val_r = self.predict(data_loader.X_val, data_loader.y_val,no_pred=no_pred)
        return train_r, val_r