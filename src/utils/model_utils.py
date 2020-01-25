#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import bert
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def specific_acc(label_threshold_less):
    def arg_label_acc(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        accuracy_mask = K.cast(K.less(class_id_preds,label_threshold_less),'float32')
        accuracy_mask = 1 - accuracy_mask
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'float32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return arg_label_acc

def single_class_accuracy(interesting_class_id):
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_preds with class_id_true for recall here
        accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
        return class_acc
    return fn

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
                  metrics=[single_class_accuracy(3)])
    model.summary()
    return model
