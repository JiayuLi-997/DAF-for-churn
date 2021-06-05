#!/bin/sh -x

cd 'src/Difficulty_Flow/'
python Difficulty_fitting.py --save_path ../../data/Difficulty_Flow

cd '../Data_preparation/'
python Cox-Feature-Extraction.py --data_path ../../data/ --params_file ../../data/Difficulty_Flow/0_difficulty_flow.json --save_path ../../data/D-Cox-Time --save_day_features_path ../../data/
cd '../D-Cox-Time/'
python Cox_train.py --random_seed 2021 --weight_decay 0.8 --lr 0.01 --distance_level 10 --max_session_length 30 --earlystop_patience 10 --optimizer AdamWR --cro：:ss_validation 5 --one_hot 1 --data_path ../../data/D-Cox-Time/ --fold_define ../../data/dataset_split --device cuda --model_name D-Cox-Time

cd '../Data_preparation/'
python Churn-Feature-Extraction.py --data_path "../../data/" --cox_feature_path "../D-Cox-Time/Checkpoints/fold-XXX/D-Cox-Time" --save_path "../../data/Churn-Features/"

cd '../Churn_prediction/'
# Example
## LR
python main.py --model_name LR --datapath ../../data --feature_file Churn_features/feature_data.csv --user_path dataset_split
python main.py --model_name LR --datapath ../../data --feature_file Churn_features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name LR --datapath ../../data --feature_file Churn_features/feature_data_all.csv --user_path dataset_split

## SVM
python main.py --model_name SVM --C 10.0 --kernel rbf --gamma -1.0 --datapath ../../data --feature_file Churn_features/feature_data.csv --user_path dataset_split
python main.py --model_name SVM --C 10.0 --kernel rbf --gamma -1.0 --datapath ../../data --feature_file Churn_features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name SVM --C 10.0 --kernel rbf --gamma -1.0 --datapath ../../data --feature_file Churn_features/feature_data_all.csv --user_path dataset_split

## MLP
python main.py --model_name MLP --embed_size 256 --hidden_size 1 --epoches 800 --lr 0.005 --datapath ../../data --feature_file Churn_features/feature_data.csv --user_path dataset_split
python main.py --model_name MLP --embed_size 64 --hidden_size 1 --epoches 500 --lr 0.001 --datapath ../../data --feature_file Churn_features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name MLP --embed_size 64 --hidden_size 1 --epoches 500 --lr 0.001 --datapath ../../data --feature_file Churn_features/feature_data_all.csv --user_path dataset_split

## DeepFM
python main.py --model_name DeepFM --dnn_hidden_units [256, 256] --l2_reg_linear 1e-05 --l2_reg_dnn 1e-4 --dnn_dropout 0.5 --dnn_use_bn 0 --lr 0.001 --datapath ../../data --feature_file Churn_features/feature_data.csv --user_path dataset_split
python main.py --model_name DeepFM --dnn_hidden_units [256, 256] --l2_reg_linear 1e-4 --l2_reg_dnn 1e-4 --dnn_dropout 0.9 --dnn_use_bn 0 --lr 0.001 --datapath ../../data --feature_file Churn_features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name DeepFM --dnn_hidden_units [128, 258] --l2_reg_linear 1e-4 --l2_reg_dnn 1e-05 --dnn_dropout 0.5 --dnn_use_bn 0 --lr 0.001 --datapath ../../data --feature_file Churn_features/feature_data_all.csv --user_path dataset_split

## RF
python main.py --model_name RF --estimators 500 --subsample 1 --max_depth 12 --min_samples_split 2 --min_samples_leaf 1 --datapath ../../data --feature_file Churn_features/feature_data.csv --user_path dataset_split
python main.py --model_name RF --estimators 500 --subsample 1 --max_depth 12 --min_samples_split 2 --min_samples_leaf 1 --datapath ../../data --feature_file Churn_features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name RF --estimators 700 --subsample 1 --max_depth 14 --min_samples_split 2 --min_samples_leaf 1 --datapath ../../data --feature_file Churn_features/feature_data_all.csv --user_path dataset_split

## GBDT
python main.py --model_name GBDT --estimators 500 --subsample 0.9 --max_depth 6 --min_samples_split 2 --min_samples_leaf 1 --lr 0.3 --datapath ../../data --feature_file Churn_features/feature_data.csv --user_path dataset_split
python main.py --model_name GBDT --estimators 600 --subsample 1.0 --max_depth 6 --min_samples_split 2 --min_samples_leaf 1 --lr 0.3 --datapath ../../data --feature_file Churn_features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name GBDT --estimators 600 --subsample 0.9 --max_depth 6 --min_samples_split 2 --min_samples_leaf 1 --lr 0.25 --datapath ../../data --feature_file Churn_features/feature_data_all.csv --user_path dataset_split
