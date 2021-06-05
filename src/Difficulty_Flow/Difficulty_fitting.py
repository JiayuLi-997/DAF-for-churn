import numpy as np 
import pandas as pd
import logging
import argparse
import json
import time
import os
import sys

sys.path.append(os.path.join(os.getcwd(),".."))
from utils import utils

class Fit_model():

    def __init__(self,Neighbor_num=5):
        '''
        Fit the difficulty line for each user.
        Args:
            Neighbor_num: Number of neighbor data to smooth the line.
        '''
        self.Neighbor_num = Neighbor_num

    def plot_linear(self,x,y, a,b, uid=0,savefig=False,filename=""):
        # scatter original data, plot the linear ax+b
        plt.figure(figsize=(10,5))
        ax = plt.subplot()
        ax.scatter(x,y,s=2)
        ax.plot(x,a*x+b,color='r')
        ax.set_xlabel("level difficulty (global retry time)",fontsize=16)
        ax.set_ylabel("user's retry time",fontsize=16)
        ax.set_title("User %d"%(uid),fontsize=18)
        if savefig:
            plt.savefig("figures/"+filename)
        plt.show()

    def oneuser_fit(self,x,y):
        '''
        Args:
            x: global retry time.
            y: user retry time.
        '''
        y_cum = y.cumsum()
        Neighbor = self.Neighbor_num
        y_smooth = np.array([(y_cum[min(i+Neighbor,len(y)-1)]-y_cum[max(i-Neighbor,0)])/(2*Neighbor+1-max(Neighbor-i,0)-max(Neighbor-(len(y)-i),0)) for i in range(len(y))])
        A = np.vstack([x,np.ones(len(x))]).T
        try:
            results = np.linalg.lstsq(A,y_smooth, rcond=None)
        except:
            results = [[0,0],[0,0]]
            print("ERROR FIT: ",results)
        if len(results[1]):
            a,b,mse =  results[0][0], results[0][1], results[1][0]
        else:
            a,b = results[0][0], results[0][1]
            mse = 0
        if y_smooth.var()==0:
            r = 1
        else:
            r = 1 - mse / (y_smooth.var() * len(y_smooth))
        return a,b, mse, r
    
    def group_fit(self,win_interactions):
        '''
        Args:
            win_interactions: data of the win play. 
            (must include user_id, level_id, global_retrytime, retry_time)
        '''
        # Drop levels close to session stop / user churn
        if "session_length" in win_interactions.columns and "session_depth" in win_interactions.columns:
            win_interactions = win_interactions.loc[win_interactions.session_depth<win_interactions.session_length-1]

        if "level_id" in win_interactions.columns:
            retry_data = win_interactions.groupby(["user_id","level_id"]).head(1).copy()   # get the first time win
        else:
            retry_data = win_interactions

        new_global_retry = retry_data.groupby(["level_id"]).retry_time.mean().reset_index()
        new_global_retry.rename(columns={"retry_time":"global_retrytime"},inplace=True)
        retry_data = retry_data.merge(new_global_retry,on=["level_id"],how="left")

        retry_data.fillna(0,inplace=True)
        retry_dedup = retry_data.groupby(["user_id","global_retrytime"]).agg({"retry_time":"mean"}).reset_index()  # if two levels have the same global_retrytime

        # fit the curve
        retry_uid = set(retry_dedup.user_id.tolist())
        param_dict = {}
        selist, selist_all, rlist_all = [],[], []
        for k,uid in enumerate(retry_uid):
            x = retry_dedup.loc[retry_dedup.user_id==uid].global_retrytime.to_numpy()
            y = retry_dedup.loc[retry_dedup.user_id==uid].retry_time.to_numpy()
            x,y = zip(*sorted(zip(x,y),key=lambda a: a[0]))
            a,b, mse, r = self.oneuser_fit(np.array(x),np.array(y))
            param_dict[int(uid)] = [a,b]
            selist.append(mse/len(x))
            selist_all.append(mse)
            rlist_all.append(r)
        return param_dict, np.mean(selist),np.mean(rlist_all)

def Get_group_ab(filepath="interactions.csv",Neighbor_num=5):
    t0 = time.time()
    logging.info("Loading data...")
    Interactions = pd.read_csv(filepath,engine='python')
    # retain the wining results only
    win_interactions = Interactions.loc[Interactions.win==1].reset_index(drop=True)
    logging.info("Fitting difficulty curves...")
    model = Fit_model(Neighbor_num)
    param_dict, mean_mse, mean_r = model.group_fit(Interactions)
    logging.info("Difficulty curve fit MSE: {:.3f}, R^2: {:.3f} [{:<.2f} s]".format(mean_mse,mean_r,time.time()-t0)+os.linesep)
    return param_dict, mean_mse, mean_r

def Get_user_ab(x,y,Neighbor_num=5):
    model = Fit_model(Neighbor_num)
    a,b,mse,r = model.oneuser_fit(x,y)
    return a,b,mse,r

def add_args(parser):
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file save path')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--data_file',type=str,default='../../data/interactions.csv',
                        help='Input data path.')
    parser.add_argument('--save_path',type=str,default='../../data',
                        help='Save parameters a&b path')
    parser.add_argument('--file_prefix',type=str, default='', help="Prefix of save parameter file.")
    parser.add_argument('--Neighbor_num',type=int,default=5,
                        help='Number of neighbor y to smooth the curve.')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args, extras = parser.parse_known_args()

    if args.log_file == '':
        args.log_file = '../logs/{}/{}.txt'.format(utils.get_date(), "difficulty_flow_fit.log")

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    params, mse, r = Get_group_ab(args.data_file,args.Neighbor_num)

    if args.file_prefix == '':
        args.file_prefix = '0'

    logging.info("Saving parameters...")
    save_filename = os.path.join(args.save_path,args.file_prefix+"_difficulty_flow.json")
    utils.check_dir(save_filename)
    with open(save_filename,"w") as F: 
        json.dump(params,F)
    logging.info("All DONE!")