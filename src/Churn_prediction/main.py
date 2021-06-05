import numpy as np 
import pandas as pd 
import logging
import argparse

import sys

from Models import *
from helpers import utils, data_loader

def parse_global_args(parser):
    parser.add_argument("--test_only", type=bool, default=False, 
                       help="Whether test the dataset only")
    parser.add_argument("--note",type=str,default="test",
                        help="Note to add for log file name.")
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path')
    parser.add_argument('--random_seed', type=int, default=2020,
                        help='Random seed for all, numpy and pytorch.')
    parser.add_argument("--save_model",type=int, default=0)
    return parser

if __name__ == "__main__":
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument("--model_name",type=str, default="GBDT",
                       help="model name(LR, SVM, MLP, GBDT, DeepFM, or RF)")
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}.{0}'.format(init_args.model_name))

    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = model_name.parse_model_args(parser)
    parser = data_loader.Data_loader.parse_data_args(parser)
    args, extras = parser.parse_known_args()
    args.model_name = init_args.model_name
    
    log_args = [args.note, str(args.random_seed)]
    log_file_name = '_'.join(log_args).replace(' ', '_')
    log_file_dir = args.model_name
    
    if args.log_file == '':
        args.log_file = 'logs/{}/{}.txt'.format( log_file_dir,log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = ['check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
               'regenerate', 'sep', 'train', 'verbose']
    arg_str = utils.format_arg_str(args, exclude_lst=exclude)
    logging.info(arg_str)

    fold_mean = [[],[],[]]
    train_mean = [[],[],[]]
    for fold in range(1,6):
        # Cross validation
        user_file = [args.user_path+"/fold-%d/train.uid.npy"%(fold),args.user_path+"/fold-%d/dev.uid.npy"%(fold)]
        feature_file = args.feature_file if "all" not in args.feature_file else args.feature_file.replace("XXX",str(fold))
        train_type = "basic"
        if "diff" in args.feature_file:
            train_type = "diff"
        elif "all" in args.feature_file:
            train_type = "diff_inf"

        if args.model_name in ["DeepFM"]:
            loader = data_loader.FM_Data_loader(args.datapath,feature_file,args.label_file,user_file)
            loader.generate_data()
            classifier = model_name(args,loader)
            classifier.fit(loader.X_train,loader.y_train, loader.X_val, loader.y_val)
            no_pred=True
        else: 
            loader = data_loader.Data_loader(args.datapath, feature_file, args.label_file,user_file)
            loader.generate_data()
            classifier = model_name(args)
            classifier.fit(loader.X_train, loader.y_train)
            no_pred=False

        results = classifier.test(loader, no_pred=no_pred)

        Output = [args.model_name,str(fold), train_type, arg_str]
        for i,state in enumerate(["train","validation"]):
            logging.info("Fold {} : #{} set results:".format(fold,state))
            for j,metric in enumerate(["auc","acc","f1","precision","recall"]):
                logging.info("--# {0}: {1:.3f}".format(metric,results[i][j]))
                Output.append("{:.3f}".format(results[i][j]))

        fold_mean[0].append(results[1][0])
        fold_mean[1].append(results[1][1])
        fold_mean[2].append(results[1][2])
        train_mean[0].append(results[0][0])
        train_mean[1].append(results[0][1])
        train_mean[2].append(results[0][2])

        if args.save_model:
            model_path = os.path.join("Checkpoints",args.model_name,str(fold))
            os.makedirs(os.path.join("Checkpoints",args.model_name),exist_ok=True)
            classifier.save_model()    

        with open("All_results.csv","a") as F:
            F.write("\t".join(Output)+"\n")

    logging.info("auc: %.3f acc: %.3f f1: %.3f"%(np.mean(fold_mean[0]), np.mean(fold_mean[1]), np.mean(fold_mean[2])))
    logging.info("auc: %.3f acc: %.3f f1: %.3f"%(np.mean(train_mean[0]), np.mean(train_mean[1]), np.mean(train_mean[2])))
