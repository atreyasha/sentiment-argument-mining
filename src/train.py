#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from utils.data_utils import *
from utils.model_utils import *
from utils.arg_metav_formatter import *
from pre_process_us_election_debates import *

def pipe_tokenized_data(max_seq_length):
    label_map = {"<pad>":0,"[CLS]":1,"[SEP]":2,"N":3,"C":4,"P":5}
    train, test = corpus2tokens(max_seq_length=max_seq_length)
    return train, test, label_map

def train(max_epochs=10,max_seq_length=128):
    train_data, test_data, label_map = pipe_tokenized_data(max_seq_length)
    input_ids, label_ids, _ = project_to_ids(train_data,label_map)
    l_bert, model_ckpt = fetch_bert_layer()
    model = create_model(l_bert,model_ckpt)
    lr_scheduler = create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                  end_learn_rate=1e-7,
                                                  warmup_epoch_count=20,
                                                  total_epoch_count=max_epochs)
    model.fit(x=input_ids,y=label_ids,
            validation_split=0.15,
            batch_size=48,
            shuffle=True,
            epochs=total_epoch_count,
            callbacks=[lr_scheduler,
                        tf.keras.callbacks.EarlyStopping(patience=20,
                                                        restore_best_weights=True)])
    model.save_weights('./test.h5', overwrite=True)

# TODO test out masking layer to see how it affects certain positions
# TODO multiple with mask as integer, afterwards play with accuracy metrics

if __name__ == "__main__":
    train()

