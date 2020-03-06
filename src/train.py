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
from collections import Counter
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from utils.model_utils import *
from utils.arg_metav_formatter import *
from pre_process import *

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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

def mean_labels(input_dict):
    sum_f1 = (float(input_dict["3"]["f1-score"])+
              float(input_dict["4"]["f1-score"])+
              float(input_dict["5"]["f1-score"]))
    return sum_f1/3

def single_train(max_seq_length=512,max_epochs=100,batch_size=10,
                 warmup_epoch_count=10,max_learn_rate=1e-5,
                 end_learn_rate=1e-7,model_type="TD_Dense",
                 label_threshold_less=3):
    # read in data
    (train_X, train_Y,
     test_X, test_Y, label_map) = read_or_create_data(max_seq_length)
    num_labels = len(label_map.keys())
    # clear keras session
    tf.keras.backend.clear_session()
    # create log directory and log file
    log_dir = "./model_logs/"+getCurrentTime()+"_single_train/"
    os.makedirs(log_dir)
    with open(log_dir+"log.csv","w") as csvfile:
        fieldnames = ["id", "model_type",
                      "max_epochs", "train_epochs", "batch_size",
                      "warmup_epoch_count", "max_learn_rate", "end_learn_rate",
                      "train_f1", "test_f1", "test_f1_N", "test_f1_C",
                      "test_f1_P"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    # get bert layer
    l_bert, model_ckpt = fetch_bert_layer()
    # prepare model compilation
    model = create_model(l_bert,model_ckpt,max_seq_length,
                        num_labels,label_threshold_less,model_type)
    LRScheduler = learning_rate_scheduler(max_learn_rate=max_learn_rate,
                                          end_learn_rate=end_learn_rate,
                                          warmup_epoch_count=warmup_epoch_count,
                                          total_epoch_count=max_epochs)
    # train model
    history = model.fit(x=train_X,y=train_Y,
                        validation_split=0.15,
                        batch_size=batch_size,
                        shuffle=True,
                        epochs=max_epochs,
                        callbacks=[LRScheduler,
                        EarlyStopping(monitor="val_loss",
                                      mode="min",
                                      patience=10,
                                      restore_best_weights
                                      =True),
                        ModelCheckpoint(monitor="val_loss",
                                        mode="min",
                                        filepath=log_dir+
                                        "model_0.h5",
                                        save_best_only
                                        =True),
                        CSVLogger(filename=
                                  log_dir+
                                  "model_history_0.csv")])
    # find actual trained epochs
    train_epochs = len(history.history["val_loss"])
    # find train f1
    y_pred = model.predict(train_X)
    y_pred = np.argmax(y_pred,axis=-1)
    train_out_dict = class_report(train_Y,y_pred)
    train_f1 = mean_labels(train_out_dict)
    # find test f1
    y_pred = model.predict(test_X)
    y_pred = np.argmax(y_pred,axis=-1)
    test_out_dict = class_report(test_Y,y_pred)
    test_f1 = mean_labels(test_out_dict)
    # write to log file
    with open(log_dir+"log.csv","a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({"id":str(0), "model_type":model_type,
                         "max_epochs":str(max_epochs),
                         "train_epochs":str(train_epochs),
                         "batch_size":str(batch_size),
                         "warmup_epoch_count":str(warmup_epoch_count),
                         "max_learn_rate":str(max_learn_rate),
                         "end_learn_rate":str(end_learn_rate),
                         "train_f1":str(train_f1),
                         "test_f1":str(test_f1),
                         "test_f1_N":str(test_out_dict["3"]["f1-score"]),
                         "test_f1_C":str(test_out_dict["4"]["f1-score"]),
                         "test_f1_P":str(test_out_dict["5"]["f1-score"])})

def grid_train(max_seq_length=512,max_epochs=100,batch_size=10,
               label_threshold_less=3):
    # read in data
    (train_X, train_Y,
     test_X, test_Y, label_map) = read_or_create_data(max_seq_length)
    num_labels = len(label_map.keys())
    # create log directory and log file
    log_dir = "./model_logs/"+getCurrentTime()+"_grid_train/"
    os.makedirs(log_dir)
    with open(log_dir+"log.csv","w") as csvfile:
        fieldnames = ["id", "model_type",
                      "max_epochs", "train_epochs", "batch_size",
                      "warmup_epoch_count", "max_learn_rate", "end_learn_rate",
                      "train_f1", "test_f1", "test_f1_N", "test_f1_C",
                      "test_f1_P"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    # get bert layer
    l_bert, model_ckpt = fetch_bert_layer()
    # define grid-search dictionary
    grid = {"model_type":["TD_Dense","1D_Conv","Stacked_LSTM"],
            "learn_rate_combinations":[[1e-4,1e-5],
                                       [1e-4,1e-6]],
            "warmup_epoch_count":[10,15,20]}
    # create flat combinations
    iterable_grid = list(ParameterGrid(grid))
    # define starting test
    record_test = 0
    # start training loop
    for i,config in enumerate(iterable_grid):
        # clear keras session
        tf.keras.backend.clear_session()
        # define grid variables
        globals().update(config)
        max_learn_rate = learn_rate_combinations[0]
        end_learn_rate = learn_rate_combinations[1]
        # prepare model compilation
        model = create_model(l_bert,model_ckpt,max_seq_length,
                             num_labels,label_threshold_less,model_type)
        LRScheduler = learning_rate_scheduler(max_learn_rate=max_learn_rate,
                                              end_learn_rate=end_learn_rate,
                                              warmup_epoch_count=warmup_epoch_count,
                                              total_epoch_count=max_epochs)
        # train model
        history = model.fit(x=train_X,y=train_Y,
                            validation_split=0.15,
                            batch_size=batch_size,
                            shuffle=True,
                            epochs=max_epochs,
                            callbacks=[LRScheduler,
                                       EarlyStopping(monitor="val_loss",
                                                     mode="min",
                                                     patience=10,
                                                     restore_best_weights
                                                     =True),
                                       ModelCheckpoint(monitor="val_loss",
                                                       mode="min",
                                                       filepath=log_dir+
                                                       "model_"+str(i)
                                                       +".h5",
                                                       save_best_only
                                                       =True),
                                       CSVLogger(filename=
                                                 log_dir+"model_history_"
                                                 +str(i)+".csv")])
        # find actual trained epochs
        train_epochs = len(history.history["val_loss"])
        # find train f1
        y_pred = model.predict(train_X)
        y_pred = np.argmax(y_pred,axis=-1)
        train_out_dict = class_report(train_Y,y_pred)
        train_f1 = mean_labels(train_out_dict)
        # find test f1
        y_pred = model.predict(test_X)
        y_pred = np.argmax(y_pred,axis=-1)
        test_out_dict = class_report(test_Y,y_pred)
        test_f1 = mean_labels(test_out_dict)
        # write to log file
        with open(log_dir+"log.csv","a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({"id":str(i), "model_type":model_type,
                             "max_epochs":str(max_epochs),
                             "train_epochs":str(train_epochs),
                             "batch_size":str(batch_size),
                             "warmup_epoch_count":str(warmup_epoch_count),
                             "max_learn_rate":str(max_learn_rate),
                             "end_learn_rate":str(end_learn_rate),
                             "train_f1":str(train_f1),
                             "test_f1":str(test_f1),
                             "test_f1_N":str(test_out_dict["3"]["f1-score"]),
                             "test_f1_C":str(test_out_dict["4"]["f1-score"]),
                             "test_f1_P":str(test_out_dict["5"]["f1-score"])})
        # clear memory
        del model
        # filter out best model and history
        best_test = test_f1
        if best_test >= record_test:
            record_test = best_test
            todel = [el for el in glob(log_dir+"*") if ('_'+str(i)+'.')
                     not in el]
            if len(todel) > 0:
                for el in todel:
                    if "log.csv" not in el:
                        os.remove(el)
        else:
            os.remove(log_dir+"model_history_"+str(i)+".csv")
            os.remove(log_dir+"model_"+str(i)+".h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="maximum sequence length of tokenized id's")
    parser.add_argument("--grid-search", action="store_true", default=False,
                        help="True if grid search should be performed, "+
                        "otherwise single training session will commence")
    parser.add_argument("--max-epochs", type=int, default=100,
                        help="maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="batch-size for training procedure")
    single = parser.add_argument_group("arguments specific to single training")
    single.add_argument("--warmup-epochs", type=int, default=10,
                        help="warmup or high learning rate epochs")
    single.add_argument("--max-learn-rate", type=float, default=1e-5,
                        help="peak learning rate before exponential decay")
    single.add_argument("--end-learn-rate", type=float, default=1e-7,
                        help="final learning rate at end of planned training")
    single.add_argument("--model-type", type=str, default="TD_Dense",
                        help="top layer after albert, options are"+
                        " 'TD_Dense', '1D_CNN' or 'Stacked_LSTM'")
    args = parser.parse_args()
    if not args.grid_search:
        single_train(args.max_seq_length,args.max_epochs,
                     args.batch_size,args.warmup_epochs,
                     args.max_learn_rate,args.end_learn_rate,
                     args.model_type)
    else:
        grid_train(args.max_seq_length,args.max_epochs,
                   args.batch_size)
