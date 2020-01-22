#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import bert
import math
import tensorflow as tf

def fetch_bert_layer():
    model_name = "albert_base_v2"
    model_dir = bert.fetch_google_albert_model(model_name, ".models")
    model_ckpt = os.path.join(model_dir, "model.ckpt-best")
    model_params = bert.albert_params(model_dir)
    l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
    return l_bert, model_ckpt

def create_learning_rate_scheduler(max_learn_rate=1e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=20,
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
    return tf.keras.callbacks.LearningRateScheduler(lr_scheduler,
                                                    verbose=1)

def create_model(l_bert,model_ckpt,max_seq_len=128):
    input_ids = tf.keras.layers.Input(shape=(max_seq_len,),
                                      dtype='int32',
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
