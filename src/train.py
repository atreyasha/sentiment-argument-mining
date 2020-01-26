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

def single_train(max_seq_length=128,max_epochs=100,batch_size=48,
                 warmup_epoch_count=10,max_learn_rate=1e-5,
                 end_learn_rate=1e-7,model_type="dense_0"):
    # read in data
    label_threshold_less = 3
    (train_X, train_Y,
     test_X, test_Y, label_map) = read_or_create_data(max_seq_length)
    num_labels = len(label_map.keys())
    # clear keras session
    tf.keras.backend.clear_session()
    # create log directory and log file
    log_dir = "./model_logs/"+getCurrentTime()+"_single_train/"
    os.makedirs(log_dir)
    with open(log_dir+"log.csv","w") as csvfile:
        fieldnames = ["id", "model_type", "max_epochs", "train_epochs",
                      "batch_size", "warmup_epoch_count", "max_learn_rate",
                      "end_learn_rate", "train_f1", "test_f1"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    # get bert layer
    l_bert, model_ckpt = fetch_bert_layer()
    # prepare model compilation
    count = 0
    model = create_model(l_bert,model_ckpt,max_seq_length,
                         num_labels,label_threshold_less)
    LrScheduler = create_learning_rate_scheduler(max_learn_rate=max_learn_rate,
                                                 end_learn_rate=end_learn_rate,
                                                 warmup_epoch_count=
                                                 warmup_epoch_count,
                                                 total_epoch_count=max_epochs)
    # train model
    history = model.fit(x=train_X,y=train_Y,
              validation_split=0.15,
              batch_size=batch_size,
              shuffle=True,
              epochs=max_epochs,
              callbacks=[LrScheduler,
                         tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=10,
                                                          restore_best_weights
                                                          =True),
                         tf.keras.callbacks.ModelCheckpoint(monitor="val_loss",
                                                            filepath=log_dir+
                                                            "model_"+str(count)
                                                            +".h5",
                                                            save_best_only
                                                            =True),
                         tf.keras.callbacks.CSVLogger(filename=
                                                      log_dir+"model_history_"
                                                      +str(count)+".csv")])
    # find actual trained epochs
    train_epochs = len(history.history["val_loss"])
    # find train f1
    y_pred = model.predict(train_X)
    y_pred = np.argmax(y_pred,axis=-1)
    train_f1 = filtered_f1(train_Y,y_pred)
    # find test f1
    y_pred = model.predict(test_X)
    y_pred = np.argmax(y_pred,axis=-1)
    test_f1 = filtered_f1(test_Y,y_pred)
    # write to log file
    with open(log_dir+"log.csv","a") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({"id":str(count), "model_type":model_type,
                         "max_epochs":str(max_epochs),
                         "train_epochs":str(train_epochs),
                         "batch_size":str(batch_size),
                         "warmup_epoch_count":str(warmup_epoch_count),
                         "max_learn_rate":str(max_learn_rate),
                         "end_learn_rate":str(end_learn_rate),
                         "train_f1":str(train_f1),
                         "test_f1":str(test_f1)})

def grid_train():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--max-seq-length", type=int, default=128,
                        help="maximum sequence length of tokenized id's")
    parser.add_argument("--grid-search", action="store_true", default=False,
                        help="True if grid search should be performed, "+
                        "otherwise single training session will commence")
    single = parser.add_argument_group("arguments specific to single training")
    single.add_argument("--max-epochs", type=int, default=100,
                        help="maximum number of training epochs")
    single.add_argument("--batch-size", type=int, default=48,
                        help="batch-size for training procedure")
    single.add_argument("--warmup-epochs", type=int, default=10,
                        help="warmup or high learning rate epochs")
    single.add_argument("--max-learn-rate", type=float, default=1e-5,
                        help="peak learning rate before exponential decay")
    single.add_argument("--end-learn-rate", type=float, default=1e-7,
                        help="final learning rate at end of planned training")
    single.add_argument("--model-type", type=str, default="dense_0",
                        help="top layer after albert, options are 'dense_0',"+
                        " 'dense_1', 'cnn' or 'lstm'")
    args = parser.parse_args()
    if not args.grid_search:
        single_train(args.max_seq_length,args.max_epochs,
                     args.batch_size,args.warmup_epochs,
                     args.max_learn_rate,args.end_learn_rate,
                     args.model_type)

# TODO add log.csv with all information, make f1 calculating script to compute train and test f1s
# TODO add model types to create model
# TODO finish single run, test and extrapolate to grid search
# TODO clear session on each grid-search loop, delete models and objects to save memory
# TODO create logging system to retrieve scores and understand models -> have combined log.csv and init.csv and model-specific history log
# TODO create grid search parameters, use only albert base for simplicity, collect all possible parameters for summary, possibly with accumulation optimizer, learning rate possibilities -> use script to delete worse models
# TODO add necessary parameters as command line arguments, eg. total maximum epochs, batch size, model type, grid search or single
# think about how to handle test data and how to select best model from grid search
