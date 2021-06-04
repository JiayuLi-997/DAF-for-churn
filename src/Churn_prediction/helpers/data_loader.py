import numpy as np 
import pandas as pd 
import os
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import sys
from helpers import utils

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

class Data_loader:
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument("--datapath", type=str, default="./dataset",
                        help="Path for folder of dataset")
        parser.add_argument("--feature_file",type=str,default="feature_data.csv", 
                            help="File to save features.")
        parser.add_argument("--label_file",type=str,default="label.csv")
        parser.add_argument("--user_path", type=str, default="./dataset/user_split/")

        return parser

    def __init__(self,datapath="./dataset",feature_file="feature_data.csv", label_file = "label.csv",
                        user_file=["train.uid.npy","dev.uid.npy"]):
        self.datapath = datapath
        self.feature_file  = feature_file
        self.label_file = label_file
        self.user_file = user_file
        self.load_data()

    def load_data(self):
        self.feature_df = pd.read_csv(os.path.join(self.datapath, self.feature_file))
        label = pd.read_csv(os.path.join(self.datapath,self.label_file))
        self.label = label[["new_id","label"]].rename(columns={"new_id":"user_id"})
        self.feature_df = self.feature_df.drop(columns=["label"]).merge(self.label,on=["user_id"])

        self.uid_list = []        
        for file in self.user_file:
            self.uid_list.append(np.load(os.path.join(self.datapath,file)))

    def generate_data(self):
        drop_columns=["label"]
        if "user_id" in self.feature_df.columns:
            drop_columns.append("user_id")
        if "interval_length" in self.feature_df.columns:
            drop_columns.append("interval_length")
        
        train_df = self.feature_df.loc[self.feature_df.user_id.isin(self.uid_list[0])]
        val_df = self.feature_df.loc[self.feature_df.user_id.isin(self.uid_list[1])]

        self.X_train = train_df.drop(drop_columns, axis=1).to_numpy()
        self.y_train = train_df["label"].to_numpy()
        self.X_val = val_df.drop(drop_columns,axis=1).to_numpy()
        self.y_val = val_df["label"].to_numpy()

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)

class FM_Data_loader(Data_loader):
    
    def __init__(self,datapath="./dataset",feature_file="feature_data.csv", label_file = "label.csv",
                        user_file=["train.uid.npy","dev.uid.npy"]):
        
        super(FM_Data_loader,self).__init__(datapath=datapath,feature_file=feature_file,label_file = label_file,
                        user_file = user_file)
        self.feature_processing()

    def feature_processing(self):
        sparse_features = ["first_interval","last_interval"]
        dense_features = self.feature_df.columns.drop(["user_id","label","interval_length",
                            "first_interval","last_interval"]).tolist()
        self.feature_df[sparse_features] = self.feature_df[sparse_features].fillna('-1', )
        self.feature_df[dense_features] = self.feature_df[dense_features].fillna(0, )
        self.target = ['label']
        for feat in sparse_features:
            lbe = LabelEncoder()
            self.feature_df[feat] = lbe.fit_transform(self.feature_df[feat])
        mms = StandardScaler()
        self.feature_df[dense_features] = mms.fit_transform(self.feature_df[dense_features])
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=self.feature_df[feat].nunique(),embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,) for feat in dense_features]
        self.dnn_feature_columns = fixlen_feature_columns
        self.linear_feature_columns = fixlen_feature_columns
        self.feature_names = get_feature_names(self.linear_feature_columns)

    def generate_data(self,train_time=0, cv_fold=5):
        train_df = self.feature_df.loc[self.feature_df.user_id.isin(self.uid_list[0])]
        val_df = self.feature_df.loc[self.feature_df.user_id.isin(self.uid_list[1])]
        
        self.X_train = {name:train_df[name] for name in self.feature_names} 
        self.X_val = {name:val_df[name] for name in self.feature_names}
        self.y_train = train_df[self.target].values 
        self.y_val = val_df[self.target].values


        