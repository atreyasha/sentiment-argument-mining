#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import bert
import numpy as np
import tensorflow as tf
from obj.arg_metav_formatter import *
from tqdm import tqdm

def read_data(directory="./data/pre-processed/task_1/tokens/"):
    with open(directory+"train.pickle","rb") as f:
        train_data = pickle.load(f)
    return train_data

def project_to_ids(train_data,label_id_map,max_seq_length=300):
    model_name = "albert_base_v2"
    model_dir    = bert.fetch_google_albert_model(model_name, ".models")
    spm = "./.models/albert_base_v2/albert_base/30k-clean.model"
    vocab = "./.models/albert_base_v2/albert_base/30k-clean.vocab"
    Tokenizer = bert.albert_tokenization.FullTokenizer(vocab,
                                                       spm_model_file=spm)
    input_ids = []
    label_ids = []
    token_type_ids = []
    for instance_set in train_data:
        input_ids_sub = ["[CLS]"]
        label_ids_sub = ["[CLS]"]
        token_type_ids_sub = [0]
        for i in range(len(instance_set[1])):
            input_ids_sub.extend(instance_set[1][i])
            label_ids_sub.extend(instance_set[2][i])
            token_type_ids_sub.extend([i]*len(instance_set[1][i]))
            input_ids_sub.extend(["[SEP]"])
            label_ids_sub.extend(["[SEP]"])
            token_type_ids_sub.extend([i])
        assert (len(input_ids_sub) == len(label_ids_sub)
                == len(token_type_ids_sub))
        input_ids_sub.extend(["<pad>"]*(max_seq_length-len(input_ids_sub)))
        label_ids_sub.extend(["<pad>"]*(max_seq_length-len(label_ids_sub)))
        token_type_ids_sub.extend([i+1]*(max_seq_length-
                                         len(token_type_ids_sub)))
        assert (len(input_ids_sub) == len(label_ids_sub)
                == len(token_type_ids_sub) == max_seq_length)
        input_ids_sub = Tokenizer.convert_tokens_to_ids(input_ids_sub)
        label_ids_sub = [label_id_map[label] for label in label_ids_sub]
        input_ids.append(input_ids_sub)
        label_ids.append(label_ids_sub)
        token_type_ids.append(token_type_ids_sub)
    return np.array(input_ids), np.array(label_ids), np.array(token_type_ids)

def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):
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
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,
                                                                       verbose=1)
    return learning_rate_scheduler

def create_model(l_bert,model_ckpt,max_seq_len=300):
    """Creates a classification model."""
    input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32',
                                           name="input_ids")
    output = l_bert(input_ids)
    logits = tf.keras.layers.Dense(units=6, activation="softmax")(output)
    model = tf.keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))
    bert.load_albert_weights(l_bert, model_ckpt)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    model.summary()
    return model

train_data = read_data()
label_map = {"<pad>":0,"[CLS]":1,"[SEP]":2,"N":3,"C":4,"P":5}
input_ids, label_ids, token_type_ids = project_to_ids(train_data,label_map)

model_name = "albert_base_v2"
model_dir    = bert.fetch_google_albert_model(model_name, ".models")
model_ckpt   = os.path.join(model_dir, "model.ckpt-best")
model_params = bert.albert_params(model_dir)
l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
# TODO attempt predicting and see output, check model summary as well

model = create_model(l_bert,model_ckpt)
total_epoch_count=3

model.fit(x=input_ids,y=label_ids,
          validation_split=0.1,
          batch_size=48,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=20,
                                                    total_epoch_count=total_epoch_count),
                     tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
model.save_weights('./test.h5', overwrite=True)
