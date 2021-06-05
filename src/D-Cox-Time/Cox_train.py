import os
import sys
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import time
import json
from collections import Counter

import torch
import torchtuples as tt
from torch import nn
import torch.nn.functional as F

from pycox.models import CoxTime
from pycox.evaluation import EvalSurv

from model.Hazard_difficulty import Hazard_net
from DataLoader.Data_loader import Dataset_loader

sys.path.append(os.path.join(os.getcwd(),".."))
from utils import utils


def analyse_beta(this_beta,Scale_list,name="lose",):
    diff_level = len(Scale_list)-1
    labels = ["d in [%.2f,%.2f)"%(Scale_list[i],Scale_list[i+1]) for i in range(diff_level)]
    labels[0] = "d < %.2f"%(Scale_list[1])
    labels[-1] = "d >= %.2f"%(Scale_list[-2])
    logging.info("Range: {}".format(labels))

    Length = int(this_beta.shape[0]*0.9)
    Start, End = [0,0,int(Length/3),int(Length/3)*2],[Length,int(Length/3),int(Length/3)*2,Length]
    Range_label = ["All","First %d interactions"%(Length/3), "Middel %d interactions"%(Length/3),"Last %d interactions"%(Length/3)]
    for s,e,label in zip(Start, End,Range_label): 
        median_y = np.median(this_beta[s:e,:],axis=0)
        median_y = [round(y,3) for y in median_y]
        logging.info("Median of {} beta: {} ({})".format(label,median_y,name))
    mean = np.mean(this_beta[:Length, :], axis=0)
    mean = [round(x,3) for x in mean]
    logging.info("Mean of beta({}): {}".format(name,mean))

def parse_train_args(parser):
    parser.add_argument('--lr',type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument("--weight_decay",type=float,default=0.0)
    parser.add_argument('--batch_size',type=float, default=2048,
                        help='Batch size')
    parser.add_argument('--epochs',type=int, default=256,help='Max epochs')
    parser.add_argument('--earlystop_patience',type=int,default=10,
                        help='Tolerance epochs of early stopping, set to -1 if not use early stopping.')
    parser.add_argument("--optimizer",type=str, default="Adam")
    parser.add_argument('--cross_validation',type=int, default=0,
                        help='Whether to use cross validation.')
    parser.add_argument('--distance_level',type=int, default=5,
                        help='One-hot embedding dimension for PPD.')
    parser.add_argument('--max_window_length',type=int,default=30,
                            help='Maximum length of data to clamp.')
    parser.add_argument('--one_hot',type=int,default=0,help="Whether use one-hot encoding for other features.")
    return parser

def parse_global_args(parser):
    parser.add_argument('--data_path',type=str,default='./dataset',
                        help='Input data path.')
    parser.add_argument('--fold_define',type=str,default='./dataset',
                        help='Uids for each fold.')
    parser.add_argument('--device',type=str,default='cuda',
                        help='Device to train the model.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--train_verbose',type=int, default=1, help='Verbose while training.')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=2020,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load', type=int, default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers to load data.')
    parser.add_argument('--model_name',type=str,default="Cox_model_test")
    parser.add_argument("--fix_seed", type=int,default=0)
    return parser

def training(args,fold,random_seed,data_loader):
    # Random seed 
    if args.fix_seed:
        np.random.seed(2020)
        torch.manual_seed(2020)
        torch.cuda.manual_seed(2020)
    else:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
    
    # Read data and generate training, val, test set
    ## Define columns
    data_loader.Construct_inputdata(fold)

    x_train, u_train, y_train = data_loader.load_data("train")
    logging.info("Training data size: {}".format(x_train.shape))
    x_test, u_test, y_test = data_loader.load_data("test")
    logging.info("Test data size:{}".format(x_test.shape))

    logging.info("Defining Cox Hazard model.")
    Hazard_model = Hazard_net(feature_num=x_train.shape[1], length=x_train.shape[2]) 

    if args.device=="cuda":
        device = 0
    else:
        device = args.device
    if args.optimizer == "Adam":
        Cox_model = CoxTime(Hazard_model, tt.optim.Adam(decoupled_weight_decay=args.weight_decay),device=device, )
    elif args.optimizer == "AdamWR":
        Cox_model = CoxTime(Hazard_model, tt.optim.AdamWR(decoupled_weight_decay=args.weight_decay),device=device,)
    # training
    if args.load:
        logging.info("Loading exist model...")
        Cox_model.load_net(os.path.join("./Checkpoints/fold-%d/"%(fold),args.model_name,args.model_name+".pt"))

    utils.check_dir(os.path.join("./Checkpoints/fold-%d/"%(fold),args.model_name,args.model_name+'.pt'))
    logging.info("Training Cox hazard model...")
    if args.earlystop_patience>=0:
        callbacks = [tt.callbacks.EarlyStopping(patience=args.earlystop_patience,
                        file_path='Checkpoints/fold-%d/%s/earlystop.pt'%(fold,args.model_name))]
    else:
        callbacks = []

    Cox_model.optimizer.set_lr(args.lr)
    log = Cox_model.fit( x_train, y_train, int(args.batch_size), args.epochs, callbacks, args.train_verbose,
                       val_data=tt.tuplefy(x_test, y_test), num_workers=args.num_workers)
    logging.info("Training Done!")
    logging.info("Min val loss: {:<.4f}".format(log.to_pandas().val_loss.min()))
    beta = Cox_model.net.beta.cpu().detach().numpy()

    # save models
    logging.info("Saving best model...")
    Cox_model.save_net(os.path.join("Checkpoints/fold-%d"%(fold),args.model_name,args.model_name+".pt"))

    # calculating figures
    logging.info("Analysing difficulty hazrd parameters...")
    d_idx = data_loader.difficulty_idx
    diff_level = args.distance_level
    diff_beta = beta[d_idx["distance"]:d_idx["distance"]+diff_level,:].T
    analyse_beta(diff_beta,data_loader.difficulty_bins["distance"],"difficulty",)
    Scale_list = data_loader.difficulty_bins["distance"][1:-1]

    # test
    logging.info("Calculating performance of model on test set...")
    logging.info("Calculating baseline hazards...")
    baseline_hazards = Cox_model.compute_baseline_hazards()
    logging.info("Predicting training set...")
    surv = Cox_model.predict_surv(x_train,num_workers=args.num_workers)
    surv_df = pd.DataFrame(surv.T)
    ev = EvalSurv(surv_df, y_train[0], y_train[1], censor_surv='km')
    c_index = ev.concordance_td("antolini")
    logging.info("training set C INDEX: {:.3f}".format(c_index))
    logging.info("Predicting test set...")
    surv = Cox_model.predict_surv(x_test,num_workers=args.num_workers)
    surv_df = pd.DataFrame(surv.T)
    ev = EvalSurv(surv_df, y_test[0], y_test[1], censor_surv='km')
    c_index = ev.concordance_td("antolini")
    logging.info("test set C INDEX: {:.3f}".format(c_index))
    time_grid = np.linspace(y_test[0].min(), y_test[0].max(), x_train.shape[2]+1)
    ibs = ev.brier_score(time_grid).mean()
    logging.info("test set Brier score: {:.3f}".format(ibs))
    
    np.save(os.path.join("Checkpoints/fold-%d"%(fold),args.model_name,"beta"),beta) 
    np.save(os.path.join("Checkpoints/fold-%d"%(fold),args.model_name,"diff_beta"),diff_beta)
    np.save(os.path.join("Checkpoints/fold-%d"%(fold),args.model_name,"diff_scale"),Scale_list)
    return c_index, ibs, log, diff_beta, Scale_list

def main(args):
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # GPU
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        logging.info("# cuda devices: {}".format(torch.cuda.device_count()))

    r_list = [[],[],[]]
    logging.info("Loading data...")
    continous_features = ["level_num","play_num","duration","item_all","session_num","last_session_play","session_length",
                                    "last_session_level","last_session_item","last_session_duration","last5_duration","last5_passrate","last5_item","day_depth",
                                    "gold_amount","coin_amount"]
    categorial_features = ["weekday","last_session_end_hour","last_win","remain_energy"]
    distance_features = ["PPD"]

    data_loader = Dataset_loader(data_path=args.data_path,fold_define=args.fold_define,max_length=args.max_window_length,
                                    continous_features=continous_features,categorial_features=categorial_features,
                                    distance_features=distance_features,onehot=args.one_hot,distance_level=args.distance_level,)
    
    for k in range(max(args.cross_validation,1)):
        c_index, ibs,log, beta_diff,diff_range = training(args,fold=k+1, random_seed=args.random_seed, data_loader=data_loader)
        r_list[0].append(round(c_index,3))
        r_list[1].append(round(ibs,3))
        r_list[2].append(log.to_pandas().val_loss.min())
    logging.info("Cross validation results:")
    logging.info("C INDEX: {}".format(r_list[0]))
    logging.info("IBS: {}".format(r_list[1]))
    logging.info("Val Loss: {}".format(r_list[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    args, extras = parser.parse_known_args()

    log_args = [utils.get_date(), str(args.random_seed)]
    log_file_name = '__'.join(log_args)
    if args.log_file == '':
        args.log_file = 'logs/{}/{}.txt'.format(args.model_name, log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    Model = main(args)
