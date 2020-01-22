#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from utils.data_utils import *
from utils.model_utils import *
from utils.arg_metav_formatter import *
from pre_process_us_election_debates import *

def pipe_tokenized_data():
    label_map = {"<pad>":0,"[CLS]":1,"[SEP]":2,"N":3,"C":4,"P":5}
    train, test = corpus2tokens()
    return train, label_map

def train(max_epochs=3):
    train_data, label_map = pipe_tokenized_data()
    input_ids, label_ids, output_mask = project_to_ids(train_data,label_map)
    l_bert, model_ckpt = fetch_bert_layer()
    model = create_model(l_bert,model_ckpt)
    lr_scheduler = create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                  end_learn_rate=1e-7,
                                                  warmup_epoch_count=20,
                                                  total_epoch_count=max_epochs)
    print(model.predict(input_ids[0].reshape(1,128)))

if __name__ == "__main__":
    train()

# model.fit(x=input_ids,y=label_ids,
#           validation_split=0.1,
#           batch_size=48,
#           shuffle=True,
#           epochs=total_epoch_count,
#           callbacks=[lr_scheduler,
#                      tf.keras.callbacks.EarlyStopping(patience=20,
#                                                       restore_best_weights=True)])
# model.save_weights('./test.h5', overwrite=True)
# max_sequence_length = 128
