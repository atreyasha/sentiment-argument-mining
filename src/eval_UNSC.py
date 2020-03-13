#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model
from pre_process_UNSC import *
from utils.arg_metav_formatter import *
from utils.model_utils import *

def read_or_create_data(max_seq_length=512,
                        directory="./data/UNSC/eval/"):
    check = glob(directory+"*"+str(max_seq_length)+"*")
    if len(check) < 3:
        eval_X,_,_ = corpus2tokenids_UNSC(max_seq_length=max_seq_length)
    else:
        eval_X = np.load(directory+"eval_X_"+str(max_seq_length)+".npy")
    return eval_X

def load_saved_model(model_path):
    l_bert, model_ckpt = fetch_bert_layer()
    model = load_model(model_path,
                       custom_objects={"BertModelLayer":l_bert,
                                       "argument_candidate_acc":class_acc(3)})
    return model

def eval_model_UNSC(direct_model,max_seq_length=512,
                    direct_save="./data/UNSC/eval/"):
    eval_X = read_or_create_data(max_seq_length)
    model = load_saved_model(direct_model)
    y_pred = model.predict(eval_X,verbose=1)
    y_pred = np.argmax(y_pred,axis=-1)
    np.save(direct_save+"eval_Yhat_"+str(max_seq_length)+".npy",y_pred)

def retokenize_UNSC():
    # TODO remove all special tokens, keep only important tokens
    # project thoes back to actual tokens along with corresponding labels
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="maximum sequence length of tokenized id's")
    required = parser.add_argument_group("required name arguments")
    required.add_argument("--model-dir", required=True, type=str,
                          help="path to model *h5 file")
    args = parser.parse_args()
    eval_model_UNSC(direct_model=args.model_dir,
                    max_seq_length=args.max_seq_length)
