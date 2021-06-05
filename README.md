# DAF-for-churn
These are our datasets and implementation for the paper:

*Jiayu Li, Hongyu Lu, Chenyang Wang, Weizhi Ma, Min Zhang, Xiangyu Zhao, Wei Qi, Yiqun Liu, and Shaoping Ma, 2021. A Difficulty-Aware Framework forChurn Prediction and Intervention in Games. In KDD'21.*

Please cite our paper if you use our datasets or codes. Thanks!

```
@inproceedings{
  todo.
}
```
If you have any problem about this work or dataset, please contact with Jiayu Li (jy-li20@mails.tsinghua.edu.cn)

## Datasets
We collected anonymous data from a real-world tile-matching puzzle mobile game. The open dataset contains logs of 4089 new users in two-month interactions.

Due to the storage space limitation, our data set is uploaded to XXX.
It contains two files of user activities:

**interactions.csv**:

Each line represent a *Play* behavior of users.
```
Formatting:
user_id,session_id,level_id,win,duration,energy,retry_time,timestamp,item_all,session_depth,time
```
The `session_id` is user-specific. `session_depth` indicates how many times the user played in the same session till this records.

**payment.csv**:

Each line represent one time of purchase of users.
```
Formatting:
gold_amount,coin_amount,level,timestamp,user_id 
```
`gold` and `coin` are two different types of currency in the game. 

To provide the commercial privacy of the game company, we scale up two columns with two integers, respectively .

**level_meta.csv**:

This file is generated from the `level_id` and `retry_time` of `interactions.csv`. 
It represent the global retry time (i.e. challenge *c*) of each level. 

<br/>
If you want to use this framework for the dataset, please download the files and put them in `./data/`.
<br/>
For reproducibility, we also provide the dataset split for five-fold cross validation in offline experiments of our paper in `./data/dataset_split/`.
And the labels with observation window T=30 and detection window T=7 is in `./data/label.csv`.


## Difficulty Modeling

To run the codes, first run: `pip install -r requirements.txt`

### PPD Generation
The *Personalized Difficulty Flow* is used for generating Personalized Perceived Difficutly (PPD). Run `Difficulty_fitting.py` in `src/Difficulty_Flow` to generate the flow.

```
python Difficulty_fitting.py --save_path ../../data/Difficulty_Flow
```

### DDI Modeling
Dynamic Difficulty Influence is generated with the survival analysis model, *D-Cox-Time*. 

Feature extraction is processed with `Cox-Feature-Extraction.py` in `src/Data_preparation`.
```
python Cox-Feature-Extraction.py --data_path ../../data/ --params_file ../../data/Difficulty_Flow/0_difficulty_flow.json --save_path ../../data/D-Cox-Time --save_day_features_path ../../data/
```

The implementation of *D-Cox-Time* is in `src/D-Cox-Time`.
```
# Example
python Cox_train.py --weight_decay 0.8 --lr 0.01 --distance_level 10 --max_session_length 30 --earlystop_patience 10 --optimizer AdamWR --cross_validation 5 --one_hot 1 --data_path ../../data/D-Cox-Time/ --fold_define ../../data/dataset_split --device cuda --model_name D-Cox-Time
```


## Churn Prediction

Churn prediction is conducted with various models.

Feature extraction is processed with `Churn-Feature-Extraction.py` in `src/Data_preparation`.
```
python Churn-Feature-Extraction.py --data_path "../../data/" --cox_feature_path "../D-Cox-Time/Checkpoints/fold-XXX/D-Cox-Time" --save_path "../../data/Churn-Features/"
```

In our paper, we report the predictoin results with the best AUC. The hyper-parameters for each model and feature group in our paper are as follows:
```
# Example

## LR
python main.py --model_name LR --datapath ../../data --feature_file Churn-Features/feature_data.csv --user_path dataset_split
python main.py --model_name LR --datapath ../../data --feature_file Churn-Features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name LR --datapath ../../data --feature_file Churn-Features/feature_data_all.csv --user_path dataset_split

## SVM
python main.py --model_name SVM --C 10.0 --kernel rbf --gamma -1.0 --datapath ../../data --feature_file Churn-Features/feature_data.csv --user_path dataset_split
python main.py --model_name SVM --C 10.0 --kernel rbf --gamma -1.0 --datapath ../../data --feature_file Churn-Features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name SVM --C 10.0 --kernel rbf --gamma -1.0 --datapath ../../data --feature_file Churn-Features/feature_data_all.csv --user_path dataset_split

## MLP
python main.py --model_name MLP --embed_size 256 --hidden_size 1 --epoches 800 --lr 0.005 --datapath ../../data --feature_file Churn-Features/feature_data.csv --user_path dataset_split
python main.py --model_name MLP --embed_size 64 --hidden_size 1 --epoches 500 --lr 0.001 --datapath ../../data --feature_file Churn-Features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name MLP --embed_size 64 --hidden_size 1 --epoches 500 --lr 0.001 --datapath ../../data --feature_file Churn-Features/feature_data_all.csv --user_path dataset_split

## DeepFM
python main.py --model_name DeepFM --dnn_hidden_units [256, 256] --l2_reg_linear 1e-05 --l2_reg_dnn 1e-4 --dnn_dropout 0.5 --dnn_use_bn 0 --lr 0.001 --datapath ../../data --feature_file Churn-Features/feature_data.csv --user_path dataset_split
python main.py --model_name DeepFM --dnn_hidden_units [256, 256] --l2_reg_linear 1e-4 --l2_reg_dnn 1e-4 --dnn_dropout 0.9 --dnn_use_bn 0 --lr 0.001 --datapath ../../data --feature_file Churn-Features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name DeepFM --dnn_hidden_units [128, 258] --l2_reg_linear 1e-4 --l2_reg_dnn 1e-05 --dnn_dropout 0.5 --dnn_use_bn 0 --lr 0.001 --datapath ../../data --feature_file Churn-Features/feature_data_all.csv --user_path dataset_split

## RF
python main.py --model_name RF --estimators 500 --subsample 1 --max_depth 12 --min_samples_split 2 --min_samples_leaf 1 --datapath ../../data --feature_file Churn-Features/feature_data.csv --user_path dataset_split
python main.py --model_name RF --estimators 500 --subsample 1 --max_depth 12 --min_samples_split 2 --min_samples_leaf 1 --datapath ../../data --feature_file Churn-Features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name RF --estimators 700 --subsample 1 --max_depth 14 --min_samples_split 2 --min_samples_leaf 1 --datapath ../../data --feature_file Churn-Features/feature_data_all.csv --user_path dataset_split

## GBDT
python main.py --model_name GBDT --estimators 500 --subsample 0.9 --max_depth 6 --min_samples_split 2 --min_samples_leaf 1 --lr 0.3 --datapath ../../data --feature_file Churn-Features/feature_data.csv --user_path dataset_split
python main.py --model_name GBDT --estimators 600 --subsample 1.0 --max_depth 6 --min_samples_split 2 --min_samples_leaf 1 --lr 0.3 --datapath ../../data --feature_file Churn-Features/feature_data_diff.csv --user_path dataset_split
python main.py --model_name GBDT --estimators 600 --subsample 0.9 --max_depth 6 --min_samples_split 2 --min_samples_leaf 1 --lr 0.25 --datapath ../../data --feature_file Churn-Features/feature_data_all.csv --user_path dataset_split

```

</br>
Run `src/churn_prediction.sh` to generate all data and get the results.


