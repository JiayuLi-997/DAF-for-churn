import pandas as pd 
import numpy as np 
import json
import argparse
import logging
import time
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.getcwd(),".."))
from utils import utils

class Data_transformer():
    def __init__(self):
        pass

    def read_feature(self,data_path):
        self.raw_features = pd.read_csv(os.path.join(data_path,"interactions.raw.csv"))
        self.level_meta = pd.read_csv(os.path.join(data_path,"level_meta.csv"))
        self.payment_data = pd.read_csv(os.path.join(data_path,"payment.csv"))

    def read_params(self,filepath):
        # Difficulty Flow params
        with open(filepath) as F:
            params = json.load(F)
        self.param_dict = {}
        for key in params:
            self.param_dict[int(key)] = params[key]

    def generate_features(self):
        logging.info("Dealing with raw data...")
        # sort by time
        self.raw_features.sort_values(by=["user_id","timestamp"],inplace=True)
        self.raw_features["time"] = self.raw_features["timestamp"].map(lambda x: datetime.fromtimestamp(x))
        self.raw_features["date"] = self.raw_features["time"].map(lambda x: (x- timedelta(hours=4)).date())
        
        # Turn the date into the length from the login day
        start_date = datetime(2020,1,1)
        self.raw_features["date_interval"] = self.raw_features.date.map(lambda x: (x-start_date.date()).days)
        register_day = self.raw_features.groupby("user_id").head(1).loc[:,["user_id","date_interval"]]
        register_day.rename(columns={"date_interval":"register_date"},inplace=True)
        self.raw_features = self.raw_features.merge(register_day,on=["user_id"],how='left')
        self.raw_features["day_since_register"] = self.raw_features["date_interval"] - self.raw_features["register_date"]

        # length of each session
        session_length = self.raw_features.groupby(["user_id","date","session_id"]) \
                            .session_depth.nunique().reset_index().rename(columns={"session_depth":"session_length"})

        # day level features
        logging.info("Generating day-level features...")
        self.raw_features["cnt"] = 1
        day_features = self.raw_features.groupby(["user_id","date"]).agg(
                {"level_id":"nunique","cnt":"count","duration":"sum","item_all":"sum","session_id":"nunique"}).reset_index().rename(
                        columns={"level_id":"level_num","session_id":"session_num","cnt":"play_num"})
        day_session = session_length.groupby(["user_id","date"]).session_length.mean().reset_index()
        day_features = day_features.merge(day_session, on=["user_id","date"],how="left")
        day_features["weekday"] = day_features["date"].apply(lambda x: x.weekday())   

        # Difficulty-related features: level global retry time(challenge)，user retry time( user effort )
        logging.info("Generating difficulty-related features...")
        global_retry =  self.level_meta.loc[:,["level_id","challenge"]]
        global_retry.rename(columns={"challenge":"global_retrytime"},inplace=True)
        difficulty_level = self.raw_features.groupby(["user_id","date","level_id"]).agg({"retry_time":"max"}).reset_index()
        difficulty_level = difficulty_level.merge(global_retry, on=["level_id"],how='left')
        difficulty_level.fillna(0,inplace=True)
        day_difficulty_features = difficulty_level.groupby(["user_id","date"]).agg(
            {"retry_time":"mean","global_retrytime":"mean"}).reset_index()    
        
        self.raw_features = self.raw_features.merge(global_retry,on=["level_id"],how='left').fillna(0)

        # last session features
        logging.info("Generating features in the last session of day...")
        self.raw_features["cnt"] = 1
        session_features = self.raw_features.groupby(["user_id","session_id"]).agg(
            {"time":["min","max"],"cnt":"count","level_id":"nunique","item_all":"sum","date":"max","retry_time":"mean",
            "global_retrytime":"mean"}).reset_index()
        session_features.columns = ["_".join(c) if len(c[1]) else c[0] for c in session_features.columns ]
        session_features.rename(columns={"time_min":"start_time","time_max":"end_time","cnt_count":"play_num",
                                        "level_id_nunique":"level_num","item_all_sum":"item_all","date_max":"date"},inplace=True)
        session_features["duration"] = (session_features["end_time"] - session_features["start_time"]).apply(lambda x: x.seconds)
        session_features["start_hour"] = session_features["start_time"].apply(lambda x: x.hour)
        session_features["end_hour"] = session_features["end_time"].apply(lambda x: x.hour)
        day_session_features = session_features.groupby(["user_id","date"]).tail(1).loc[:,["user_id","play_num","level_num","item_all","date","duration","end_hour"]].rename(
                columns={"play_num":"last_session_play","level_num":"last_session_level","item_all":"last_session_item",
                        "duration":"last_session_duration","end_hour":"last_session_end_hour","retry_time":"last_session_retry",
                        "global_retrytime":"last_session_globalretry"})
        
        # last k play features
        logging.info("Generating last 5 play features...")
        k = 5
        last_k = self.raw_features.groupby(["user_id","date"]).tail(k).reset_index()
        last_k_features = last_k.groupby(["user_id","date"]).agg({"duration":"mean","win":"mean","item_all":"sum"}).reset_index().rename(
            columns={"duration":"last%d_duration"%(k),"win":"last%d_passrate"%(k),"item_all":"last%d_item"%(k)})
        last_feature = self.raw_features.groupby(["user_id","date"]).tail(1).reset_index().loc[:,["user_id","date","win","energy"]].rename(
            columns={"win":"last_win","energy":"remain_energy"})
        
        # payment feature
        logging.info("Gnerating payment features...")
        self.payment_data["time"] = self.payment_data["timestamp"].apply(lambda x: datetime.fromtimestamp(x))
        self.payment_data["date"] = self.payment_data["time"].apply(lambda x: (x- timedelta(hours=4)).date()) 
        payment_features = self.payment_data.groupby(["user_id","date"]).agg({"gold_amount":"sum","coin_amount":"sum"}).reset_index()

        day_since_register = self.raw_features.groupby(["user_id","date"]).agg({"day_since_register":"mean"})
        day_since_register.reset_index(inplace=True)

        highest_level_day = self.raw_features.groupby(["user_id","day_since_register"]).level_id.max().reset_index()
        highest_level_day.rename(columns={"level_id":"highest_level"},inplace=True)

        # merge all features
        logging.info("Merging all features ...")
        all_features = day_features.merge(day_session_features,on=["user_id","date"],how="left")
        all_features = all_features.merge(last_k_features,on=["user_id","date"],how="left")
        all_features = all_features.merge(last_feature,on=["user_id","date"],how="left")
        all_features = all_features.merge(day_difficulty_features,on=["user_id","date"],how='left')
        all_features = all_features.merge(payment_features,on=["user_id","date"],how='left')
        all_features = all_features.merge(day_since_register,on=["user_id","date"],how="left")
        all_features = all_features.merge(highest_level_day,on=["user_id","day_since_register"],how="left")
        all_features.fillna(0,inplace=True)

        self.day_features = all_features
        # fill the days without login as zeros
        churn_date = dict(all_features.groupby("user_id").day_since_register.max())
        enter_date = dict(all_features.groupby("user_id").day_since_register.min())
        rows = (np.array(list(churn_date.values()))-np.array(list(enter_date.values()))+1)
        feature_list = ['level_num','play_num', 'duration', 'item_all', 'session_num','session_length', 'weekday', 'last_session_play', 'last_session_level',
                    'last_session_item', 'last_session_duration', 'last_session_end_hour','last5_duration', 'last5_passrate', 'last5_item', 'last_win',
                    'remain_energy', 'retry_time', 'global_retrytime', 'day_depth','gold_amount','coin_amount']
        feature_idx = dict(zip(feature_list,range(len(feature_list)) ) )
        X_features = np.zeros((rows.sum(),len(feature_list)))
        User_list = np.zeros(rows.sum())
        idx = 0
        idx_np = 0
        features_np = all_features.rename(columns={"day_since_register":"day_depth"}).loc[:,feature_list].to_numpy()
        for uid in all_features.user_id.unique():
            for i in range(churn_date[uid]-enter_date[uid]+1):
                User_list[idx] = uid
                if features_np[idx_np,feature_idx["day_depth"]] == i: 
                    X_features[idx,:] = features_np[idx_np,:]
                    idx_np += 1
                else:
                    X_features[idx,feature_idx["day_depth"]] = i
                idx += 1
        interactions_np = np.concatenate((User_list.reshape(-1,1),X_features),axis=1)
        self.interactions = pd.DataFrame(interactions_np,columns=["user_id"]+feature_list)

    def difficulty_process(self):
        df = self.interactions
        # transfer absolute difficulty to personalized perceived difficulty
        X_features = []
        user_list = []
        user = df["user_id"].to_numpy()
        x = df["global_retrytime"].to_numpy()
        y = df["retry_time"].to_numpy()
        params = np.array([self.param_dict.get(int(u),[0,0]) for u in user])
        d = y - x*params[:,0] - params[:,1]
        df["PPD"] = d
        df_part = df.drop(columns=["retry_time","global_retrytime","user_id"])
        X_features = X_features + list(df_part.to_numpy())
        user_list = user_list + df["user_id"].tolist()
        self.feature_list = df_part.columns.to_numpy()        
        self.X_features_np = np.array(X_features)
        self.user_list = np.array(user_list)

    def save_dayfile(self,filepath):
        self.day_features.to_csv(os.path.join(filepath,"day_features.csv"),index=False) # day features are used for churn prediction

    def save_npfile(self,filepath):
        utils.check_dir(os.path.join(filepath,"X_features"))
        np.save(os.path.join(filepath,"X_features"),self.X_features_np)
        np.save(os.path.join(filepath,"User_list"),self.user_list)
        np.save(os.path.join(filepath,"feature_list"),self.feature_list)

def add_args(parser):
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file save path')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument("--data_path",type=str,default="./dataset/",
                        help="Data path for raw features, level metadata, and payment data.")
    parser.add_argument("--params_file",type=str,default="./dataset/difficulty_flow.json",
                         help="data filename of a&b parameters.")
    parser.add_argument('--save_path',type=str,default='./dataset',
                        help='Save X_features, User_list and feature_list dir path')
    parser.add_argument('--save_day_features_path',type=str,default='./dataset',
                        help='Save features for each day. (used for churn prediction)')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args, extras = parser.parse_known_args()

    if args.log_file == '':
        args.log_file = 'logs/{}/{}.txt'.format(utils.get_date(), "data_transform.log")

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info("Start transformation!")
    t0 = time.time()
    transformer = Data_transformer()
    logging.info("Reading data...")
    transformer.read_feature(args.data_path)
    transformer.read_params(args.params_file)
    logging.info("Generating features...")
    transformer.generate_features()
    logging.info("Processing transformation...")
    transformer.difficulty_process()
    logging.info("Saving data...")
    transformer.save_npfile(args.save_path)
    transformer.save_dayfile(args.save_day_features_path)

    logging.info("All DONE! [{:<.2f} s]".format(time.time()-t0))
