#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
import datetime
import numpy as np
import tensorflow as tf
from collections import Counter
from glob import glob
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from utils.model_utils import *
from pre_process import *

def read_or_create_data(max_seq_length,
                        directory="./data/USElectionDebates/task_1/"):
    check = glob(directory+"*"+str(max_seq_length)+"*")
    if len(check) < 4:
        (train_X, train_Y, test_X,
         test_Y, label_map) = corpus2tokenids(max_seq_length=max_seq_length)
    else:
        train_X = np.load(directory+"train_X_"+str(max_seq_length)+".npy")
        train_Y = np.load(directory+"train_Y_"+str(max_seq_length)+".npy")
        test_X = np.load(directory+"test_X_"+str(max_seq_length)+".npy")
        test_Y = np.load(directory+"test_Y_"+str(max_seq_length)+".npy")
        with open(directory+"label_map.json","r") as f:
            label_map = json.load(f)
    return train_X, train_Y, test_X, test_Y, label_map

def report(y_true,y_pred,greater_than_equal=3):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    relevant_indices = np.where(y_true >= greater_than_equal)[0]
    y_pred = y_pred[relevant_indices]
    y_true = y_true[relevant_indices]
    return classification_report(y_true,y_pred,output_dict=True)

def quick_eval(direct,max_seq_length=128):
    # read in data
    (train_X, train_Y,
     test_X, test_Y, label_map) = read_or_create_data(max_seq_length)
    num_labels = len(label_map.keys())
    # clear keras session
    tf.keras.backend.clear_session()
    # get bert layer
    l_bert, model_ckpt = fetch_bert_layer()
    model_path = glob(direct+"/*h5")
    model = load_model(model_path,
                       custom_objects={"BertModelLayer":l_bert,
                                       "argument_candidate_acc":class_acc(3)})
    y_pred = model.predict(test_X)
    y_pred = np.argmax(y_pred,axis=-1)
    out_dict = report(test_Y,y_pred)
    with open(direct+"/report.json","w") as f:
        json.dump(out_dict,f)

logs = glob("./model_logs/*")
[quick_eval(el) for el in logs]
