# -*- coding: UTF-8 -*-

import os
import logging
import torch
import datetime
import numpy as np

def format_arg_str(args, exclude_lst, max_len=20):
    linesep = os.linesep
    arg_dict = vars(args)
    keys = [k for k in arg_dict.keys() if k not in exclude_lst]
    values = [arg_dict[k] for k in keys]
    res_str = ""
    for k,v in zip(keys, values):
        res_str += '--'+str(k)+' '+str(v)+' '
    return res_str

def check_dir(file_name):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        print('make dirs:', dir_path)
        os.makedirs(dir_path)

def get_time():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")