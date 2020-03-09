#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import bert
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.layers import Conv1D, LSTM, TimeDistributed, Input
from sklearn.metrics import classification_report

def class_acc(label_threshold_less):
    def argument_candidate_acc(y_true, y_pred):
        class_id_true = K.cast(y_true,'int64')
        class_id_preds = K.argmax(y_pred, axis=-1)
        accuracy_mask = K.cast(K.less(class_id_preds,label_threshold_less),
                               'float32')
        accuracy_mask = 1 - accuracy_mask
        class_acc_tensor = (K.cast(K.equal(class_id_true,
                                          class_id_preds),
                                   'float32') * accuracy_mask)
        class_accuracy = (K.sum(class_acc_tensor)/
                          K.maximum(K.sum(accuracy_mask),1))
        return class_accuracy
    return argument_candidate_acc

def class_report(y_true,y_pred,greater_than_equal=3):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    relevant_indices = np.where(y_true >= greater_than_equal)[0]
    y_pred = y_pred[relevant_indices]
    y_true = y_true[relevant_indices]
    return classification_report(y_true,y_pred,output_dict=True)

def fetch_bert_layer():
    model_name = "albert_base_v2"
    model_dir = bert.fetch_google_albert_model(model_name, ".models")
    model_ckpt = os.path.join(model_dir, "model.ckpt-best")
    model_params = bert.albert_params(model_dir)
    l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
    return l_bert, model_ckpt

def learning_rate_scheduler(max_learn_rate,
                            end_learn_rate,
                            warmup_epoch_count,
                            total_epoch_count):
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/
                                                   max_learn_rate)*
                                          (epoch-warmup_epoch_count+1)/
                                          (total_epoch_count-
                                           warmup_epoch_count+1))
        return float(res)
    return LearningRateScheduler(lr_scheduler,verbose=1)

def create_model(l_bert,model_ckpt,max_seq_len,num_labels,
                 label_threshold_less,model_type):
    input_ids = Input(shape=(max_seq_len,),
                      dtype='int32')
    output = l_bert(input_ids)
    if model_type == "TD_Dense":
        output = TimeDistributed(Dense(512))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(256))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(128))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(64))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(num_labels))(output)
    elif model_type == "1D_CNN":
        output = Conv1D(512,3,padding="same")(output)
        output = Activation("relu")(output)
        output = Conv1D(256,3,padding="same")(output)
        output = Activation("relu")(output)
        output = Conv1D(128,3,padding="same")(output)
        output = Activation("relu")(output)
        output = Conv1D(64,3,padding="same")(output)
        output = Activation("relu")(output)
        output = Conv1D(num_labels,3,padding="same")(output)
    elif model_type == "Stacked_LSTM":
        output = LSTM(512,return_sequences=True)(output)
        output = LSTM(256,return_sequences=True)(output)
        output = LSTM(128,return_sequences=True)(output)
        output = TimeDistributed(Dense(64))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(num_labels))(output)
    prob = Activation("softmax")(output)
    model = tf.keras.Model(inputs=input_ids, outputs=prob)
    model.build(input_shape=(None, max_seq_len))
    bert.load_albert_weights(l_bert, model_ckpt)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[class_acc(label_threshold_less)])
    model.summary()
    return model
