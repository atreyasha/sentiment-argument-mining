#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
import datetime
import numpy as np
import tensorflow as tf
from glob import glob
from utils.model_utils import *
from utils.arg_metav_formatter import *
from pre_process import *

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def read_or_create_data(max_seq_length,
                        directory="./data/USElectionDebates/task_1/"):
    check = glob(directory+"*"+str(max_seq_length)+"*")
    if len(check) != 4:
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

def single_train(max_seq_length=128,max_epochs=2,batch_size=48,
                 warmup_epoch_count=10,max_learn_rate=1e-5,
                 end_learn_rate=1e-7):
    # read in data
    label_threshold_less = 3
    train_X, train_Y, _, _, label_map = read_or_create_data(max_seq_length)
    num_labels = len(label_map.keys())
    # clear keras session
    tf.keras.backend.clear_session()
    # create log directory
    log_dir = "./model_logs/"+getCurrentTime()+"_single_train/"
    os.makedirs(log_dir)
    # prepare model compilation
    l_bert, model_ckpt = fetch_bert_layer()
    model = create_model(l_bert,model_ckpt,max_seq_length,
                         num_labels,label_threshold_less)
    LrScheduler = create_learning_rate_scheduler(max_learn_rate=max_learn_rate,
                                                 end_learn_rate=min_learn_rate,
                                                 warmup_epoch_count=
                                                 warmup_epoch_count,
                                                 total_epoch_count=max_epochs)
    # train model
    model.fit(x=train_X,y=train_Y,
              validation_split=0.15,
              batch_size=batch_size,
              shuffle=True,
              epochs=max_epochs,
              callbacks=[LrScheduler,
                         tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=5,
                                                          restore_best_weights
                                                          =True),
                         tf.keras.callbacks.ModelCheckpoint(monitor="val_loss",
                                                            filepath=log_dir+
                                                            "best_model.h5",
                                                            save_best_only=True)])

def grid_train():
    pass

if __name__ == "__main__":
    single_train()

# TODO finish single run, test and extrapolate to grid search
# TODO clear session on each grid-search loop
# TODO add checkpoint callback for saving models, need to generate separate folder
# TODO create logging system to retrieve scores and understand models -> have combined log.csv and init.csv and model-specific history log
# TODO create grid search parameters, use only albert base for simplicity, collect all possible parameters for summary, possibly with accumulation optimizer, learning rate possibilities -> use script to delete worse models
# TODO add necessary parameters as command line arguments, eg. total maximum epochs, batch size, model type, grid search or single
# think about how to handle test data and how to select best model from grid search
