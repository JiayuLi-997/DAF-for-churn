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
For reproducibility, we also provide the dataset split for five-fold cross validation in offline experiments of our paper in `./data/dataset_split`.


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
python Cox_train.py --random_seed 2021 --weight_decay 0.8 --lr 0.01 --distance_level 10 --max_session_length 30 --earlystop_patience 10 --optimizer AdamWR --cro：:ss_validation 5 --one_hot 1 --data_path ../../data/D-Cox-Time/ --fold_define ../../data/dataset_split --device cuda --model_name D-Cox-Time
```


## Churn Prediction



