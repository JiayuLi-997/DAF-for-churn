import os
import pandas as pd
import logging
import argparse
import numpy as np

from deepctr_torch import models
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch import callbacks

from Models.base_model import base_model
from helpers import utils

import torch

class DeepFM(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--max_epoch",type=int, default=100, help="Max iterations for training.")
        parser.add_argument("--device",type=str,default='cpu')
        parser.add_argument("--batch_size",type=int,default=256)
        parser.add_argument('--earlystop_patience',type=int,default=50,
                        help='Tolerance epochs of early stopping, set to -1 if not use early stopping.')
        parser.add_argument('--dnn_hidden_units',nargs='+',type=int,help='The layer number and units in each layer of DNN.')
        parser.add_argument('--l2_reg_linear',type=float,default=1e-05)
        parser.add_argument('--l2_reg_dnn',type=float,default=0)
        parser.add_argument('--dnn_dropout',type=float,default=0)
        parser.add_argument('--dnn_activation',type=str,default='relu')
        parser.add_argument('--dnn_use_bn',type=int,default=0)
        return base_model.parse_model_args(parser)
    
    def __init__(self, args, data_loader):
        use_bn = False if args.dnn_use_bn == 0 else True
        self.classifier = models.DeepFM(data_loader.linear_feature_columns, data_loader.dnn_feature_columns,
            dnn_hidden_units=args.dnn_hidden_units, l2_reg_linear = args.l2_reg_linear, l2_reg_dnn = args.l2_reg_dnn,
            dnn_dropout = args.dnn_dropout, dnn_activation = args.dnn_activation, dnn_use_bn = use_bn,
            task='binary',device=args.device, seed=args.random_seed)
        optimizer = torch.optim.Adam(self.classifier.parameters(),lr=args.lr)
        # self.classifier.compile(optimizer,"binary_crossentropy",metrics=["binary_crossentropy","auc"],val_metrics=["auc","binary_crossentropy"])
        self.classifier.compile('adam',"binary_crossentropy",metrics=["binary_crossentropy","auc"],val_metrics=["auc","binary_crossentropy"])
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.earlystop_patience = args.earlystop_patience
        

    def fit(self,X,y,val_X,val_y):
        model_callback = []
        model_callback = [callbacks.EarlyStopping(patience=self.earlystop_patience,monitor='val_binary_crossentropy',mode="min")]
        # model_callback = [callbacks.EarlyStopping(patience=self.earlystop_patience,monitor='val_auc',mode="max")]
        
        self.classifier.fit(X,y,batch_size=self.batch_size,epochs=self.max_epoch,validation_data=(val_X,val_y),
                callbacks=model_callback,verbose=0)
    
    def model_predict(self,X):
        return self.classifier.predict(X)
    