#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from glob import glob
from utils.data_utils import *
from utils.metrics_utils import *
from utils.model_utils import *
from utils.arg_metav_formatter import *
from pre_process_us_election_debates import *

def pipe_tokenized_data(max_seq_length):
    train, test = corpus2tokens(max_seq_length=max_seq_length)
    return train, test

def create_model(l_bert,model_ckpt,max_seq_len,num_labels,
                 label_threshold_less):
    input_ids = tf.keras.layers.Input(shape=(max_seq_len,),
                                      dtype='int32',
                                      name="input_ids")
    output = l_bert(input_ids)
    logits = tf.keras.layers.Dense(units=num_labels,
                                   activation="softmax")(output)
    model = tf.keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))
    bert.load_albert_weights(l_bert, model_ckpt)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits
                                                                     =True),
                  metrics=[class_acc(label_threshold_less)])
    model.summary()
    return model

def train(max_epochs=10,max_seq_length=128,
          directory="./data/USElectionDebates/task_1/"):
    label_map = {"<pad>":0,"[CLS]":1,"[SEP]":2,"N":3,"C":4,"P":5}
    num_labels = len(label_map.keys())
    label_threshold_less = 3
    check = glob(directory+"*"+str(max_seq_length)+"*")
    if len(check) != 4:
        train_data, test_data = pipe_tokenized_data(max_seq_length)
        train_X, train_Y, _ = project_to_ids(train_data,label_map)
        test_X, test_Y, _ = project_to_ids(test_data,label_map)
        np.save(directory+"train_X_"+str(max_seq_length)+".npy",train_X)
        np.save(directory+"train_Y_"+str(max_seq_length)+".npy",train_Y)
        np.save(directory+"test_X_"+str(max_seq_length)+".npy",test_X)
        np.save(directory+"test_Y_"+str(max_seq_length)+".npy",test_Y)
        del test_X
        del test_Y
    else:
        train_X = np.load(directory+"train_X_"+str(max_seq_length)+".npy")
        train_Y = np.load(directory+"train_Y_"+str(max_seq_length)+".npy")
    l_bert, model_ckpt = fetch_bert_layer()
    model = create_model(l_bert,model_ckpt,max_seq_length,
                         num_labels,label_threshold_less)
    LrScheduler = create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                  end_learn_rate=1e-7,
                                                  warmup_epoch_count=10,
                                                  total_epoch_count=max_epochs)
    model.fit(x=train_X,y=train_Y,
              validation_split=0.15,
              batch_size=48,
              shuffle=True,
              epochs=max_epochs,
              callbacks=[LrScheduler,
                         FilteredF1Score()
                         tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                          patience=20,
                                                          restore_best_weights
                                                          =True)])
    model.save_weights('./test.h5', overwrite=True)

if __name__ == "__main__":
    train()
