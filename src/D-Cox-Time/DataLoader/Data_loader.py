import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import time
from collections import Counter

import torch
import torchtuples as tt
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn_pandas import DataFrameMapper
import joblib
import gc


class Dataset_loader():
    # Load data from dataset
    def __init__(self,data_path,fold_define, max_length=160,continous_features=[],categorial_features=[],distance_features=[],
                    onehot=False,distance_level=5,onehot_dim=5,):
        '''
        Args:
            data_path: 数据集所在路径
            max_length: session长度限制
            continous_features: 连续的feature list
            categorial_features: 离散的feature list
            distance_features: difficulty distance feature
            onehot: 是否对协变量做onehot embedding
            distance_level: 将distance划分成几个区间
            onehot_dim: 协变量one hot编码的维度，仅当onehot=True时生效
        '''
        self.data_path = data_path
        self.fold_define = fold_define
        self.max_length = max_length
        self.continous_features = continous_features
        self.categorial_features = categorial_features
        self.distance_features = distance_features
        self.onehot = onehot
        self.bins = onehot_dim
        self.distance_level = distance_level
        self._read_data()

    def Construct_sessiondata(self, fold=1): 
        # construct session (training data)   
        self.t0 = time.time()
        train = np.load(os.path.join(self.fold_define,"fold-%d/train.uid.npy"%(fold)))
        test = np.load(os.path.join(self.fold_define,"fold-%d/dev.uid.npy"%(fold)))
        # test = np.load(os.path.join(self.fold_define,"test.uid.npy"))
        self._get_dataset_idx(train, test)
        self._standardize(bins=self.bins,diff_level=self.distance_level)
        logging.info('Consturction done! [{:<.2f} s]'.format(time.time() - self.t0) + os.linesep)

    def load_data(self, dataset="train"):
        # construct session-format data for training
        logging.info("Generating session data for {} set...".format(dataset))
        if dataset=="train":
            return self._generate_session_data(self.x_train_encode, self.User_list[self.dataset_idx[0]],
                         self.session_start[0],self.session_end[0])
        elif dataset=="test":
            return self._generate_session_data(self.x_test_encode, self.User_list[self.dataset_idx[1]],
                         self.session_start[1],self.session_end[1])
        else:
            logging.info("Dataset unknown: {}".format(dataset))
            return None

    def _read_data(self):   
        # read data 
        logging.info("Loading data from \"{}\" ...".format(self.data_path))
        self.X_features = np.load(os.path.join(self.data_path,"X_features.npy"))
        self.User_list = np.load(os.path.join(self.data_path,"User_list.npy"))
        feature_list = np.load(os.path.join(self.data_path,"feature_list.npy"),allow_pickle=True) 
        self.feature_idx = dict(zip(feature_list,range(len(feature_list)) ) )
        self.start_idx = np.where(self.X_features[:,self.feature_idx["day_depth"]]==0)[0]
        self.start_idx = np.append(self.start_idx, self.X_features.shape[0])
        uid_set = set(self.User_list)
        self.uid_dict = dict(zip(sorted(list(uid_set)),range(len(uid_set))))

    def _get_dataset_idx(self,train_list,  test_list,):
        # split dataset to get train,  and test sets
        logging.info("Splitting dataset ...")
        dataset_idx = [[],[]]
        self.session_start = [[],[]]
        self.session_end = [[], []]

        def add_data(dataset_type=0, i=0):
            self.session_start[dataset_type].append(len(dataset_idx[dataset_type]))
            dataset_idx[dataset_type] += list(range(self.start_idx[i],self.start_idx[i+1]))
            self.session_end[dataset_type].append(len(dataset_idx[dataset_type]))

        TRAIN, TEST = 0,1

        for i,idx in enumerate(self.start_idx[:-1]):
            if self.User_list[idx] in train_list:
                add_data(dataset_type=TRAIN, i=i)
            else:
                add_data(dataset_type=TEST, i=i)

        self.dataset_idx =  dataset_idx
        self.x_train = self.X_features[dataset_idx[0],:]
        self.x_test = self.X_features[dataset_idx[1],:]

    def _standardize(self,bins=5, diff_level = 5):
        logging.info("Standardize features ...")
        
        if self.onehot==True:
            category_len = {}
            category_bins = []
            for feature in self.categorial_features:
                idx = self.feature_idx[feature]
                f_bins = sorted(list(set(self.x_train[:,idx])))
                f_len = len(f_bins)
                if f_len>30:
                    print("Category feature is too long: %s, %d"%(feature,f_len))
                    continue
                category_len[feature]= f_len # calculate number of category
                category_bins.append(f_bins)
            new_dim = len(self.continous_features)*(bins-1)+sum(list(category_len.values()))+diff_level*2+2
        else:
            new_dim = self.x_train.shape[1]+diff_level*2
        self.x_train_encode = np.zeros((self.x_train.shape[0],new_dim))
        self.x_test_encode = np.zeros((self.x_test.shape[0],new_dim))

        encode_id = 0
        self.feature_idx_encode = {}
        self.difficulty_idx = {}
        self.difficulty_bins = {}
        if self.onehot==False:
            self.ss,self.mm = StandardScaler(), MinMaxScaler()

        for feature in self.continous_features:
            idx = self.feature_idx[feature]
            if self.onehot==False:
                self.x_train_encode[:,encode_id] = self.ss.fit_transform(self.x_train[:,idx].reshape(-1,1)).reshape(-1)
                self.x_test_encode[:,encode_id] = self.ss.transform(self.x_test[:,idx].reshape(-1,1)).reshape(-1)
                encode_id += 1
            else:
                this_bins = bins
                train_labels,train_bins = pd.qcut(self.x_train[:,idx],q=this_bins,labels=False,retbins=True,duplicates='drop')
                test_labels = pd.cut(self.x_test[:,idx],bins=train_bins,labels=False)
                self.x_train_encode[:,encode_id:encode_id+len(train_bins)-2] = pd.get_dummies(train_labels).to_numpy()[:,:-1]
                self.x_test_encode[:,encode_id:encode_id+len(train_bins)-2] = pd.get_dummies(test_labels).to_numpy()[:,:-1]
                encode_id += len(train_bins)-2
        
        for i,feature in enumerate(self.categorial_features):
            idx = self.feature_idx[feature]
            if self.onehot==False:
                self.x_train_encode[:,encode_id] = self.mm.fit_transform(self.x_train[:,idx].reshape(-1,1)).reshape(-1)
                self.x_test_encode[:,encode_id] = self.mm.transform(self.x_test[:,idx].reshape(-1,1)).reshape(-1)
                encode_id += 1
            else:
                this_bins = category_len[feature]
                train_bins = np.array(category_bins[i])-0.01
                train_bins = np.append(train_bins, train_bins[-1]+1)
                train_labels = pd.cut(self.x_train[:,idx],bins=train_bins,labels=False)
                test_labels = pd.cut(self.x_test[:,idx],bins=train_bins,labels=False)
                label_set = set(train_labels)
                if len(label_set) != len(set(test_labels)):
                    for l in label_set:
                        if l not in test_labels:
                            for k in range(len(test_labels)):
                                if test_labels[k] in [l-1,l+1]:
                                    print("[{}] replace {} with {}".format(feature,test_labels[k],l)) 
                                    test_labels[k] = l
                                    break
                self.x_train_encode[:,encode_id:encode_id+len(train_bins)-2] = pd.get_dummies(train_labels).to_numpy()[:,:-1]
                self.x_test_encode[:,encode_id:encode_id+len(train_bins)-2] = pd.get_dummies(test_labels).to_numpy()[:,:-1]
                encode_id += len(train_bins)-2
                
        for feature in self.distance_features:
            this_bins = diff_level
            idx = self.feature_idx[feature]
            f = self.x_train[:,idx]
            _,train_bins = pd.qcut(f[np.nonzero(f)],q=this_bins,labels=False,retbins=True,duplicates='drop')
            train_labels = pd.cut(self.x_train[:,idx],bins=train_bins,labels=False)
            test_labels = pd.cut(self.x_test[:,idx],bins=train_bins,labels=False)
            def dummy(labels, value):
                encode = pd.get_dummies(labels).to_numpy()
                encode[value==0,:] = 0
                return encode
            self.x_train_encode[:,encode_id:encode_id+len(train_bins)-1] = dummy(train_labels, self.x_train[:,idx])
            self.x_test_encode[:,encode_id:encode_id+len(train_bins)-1] = dummy(test_labels, self.x_test[:,idx])
            self.difficulty_idx[feature] = encode_id
            self.difficulty_bins[feature] = train_bins
            encode_id += len(train_bins)-1
            
        del self.x_train
        del self.x_test
        gc.collect()
        self.x_train_encode = self.x_train_encode[:,:encode_id]
        self.x_test_encode = self.x_test_encode[:,:encode_id]

    def _generate_session_data(self,X_all,u_all,start_idx,end_idx):
        start_idx = np.array(start_idx)
        end_idx = np.array(end_idx)
        unsort_length = end_idx - start_idx
        idx_sort_origin = np.argsort(unsort_length)

        X_data = np.zeros((len(start_idx),X_all.shape[1],self.max_length))
        uid_list = np.zeros((len(start_idx)))
        y_length = np.zeros((len(start_idx)))
        y_censor = np.zeros((len(start_idx)))

        for i in np.arange(len(start_idx)):
            j = idx_sort_origin[i]
            s_start,s_end = start_idx[j], end_idx[j]
            s_length = s_end - s_start
            uid_list[i] = self.uid_dict[u_all[s_start]]
            if s_length <= self.max_length:
                X_data[i,:,:(s_end-s_start)] = X_all[s_start:s_end,:].T
                y_length[i] = s_length
                y_censor[i] = 1
            else:
                X_data[i,:,:] = X_all[s_start:s_start+self.max_length,:].T
                y_length[i] = self.max_length

        idx_sort = np.argsort(y_length,kind='stable')
        if (idx_sort == np.arange(0, len(idx_sort))).all():
            logging.info("Sort correct!")
            return X_data, uid_list, (y_length, y_censor)
        else:
            logging.info("Sort incorrect!")
            R = X_data[idx_sort,:,:], uid_list[idx_sort], (y_length[idx_sort],y_censor[idx_sort])
            return R
