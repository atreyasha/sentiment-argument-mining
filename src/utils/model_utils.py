#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import bert
import math
import numpy as np
import tensorflow as tf

def fetch_bert_layer():
    model_name = "albert_base_v2"
    model_dir = bert.fetch_google_albert_model(model_name, ".models")
    model_ckpt = os.path.join(model_dir, "model.ckpt-best")
    model_params = bert.albert_params(model_dir)
    l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
    return l_bert, model_ckpt

def create_learning_rate_scheduler(max_learn_rate,
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
    return tf.keras.callbacks.LearningRateScheduler(lr_scheduler,
                                                    verbose=1)
