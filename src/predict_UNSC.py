#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from glob import glob
from tensorflow.keras.models import load_model
from pre_process_UNSC import *
from utils.arg_metav_formatter import *
from utils.model_utils import *

def read_or_create_data_UNSC(max_seq_length=512,
                             directory="./data/UNSC/pred/"):
    check = glob(directory+"*"+str(max_seq_length)+"*")
    if len(check) < 3:
        (pred_tokens, pred_X,
         pred_mask)= corpus2tokenids_UNSC(max_seq_length=max_seq_length)
    else:
        with open(directory+"pred_tokens_"+str(max_seq_length)+".json","r") as f:
            pred_tokens = json.load(f)
        pred_X = np.load(directory+"pred_X_"+str(max_seq_length)+".npy")
        pred_mask = np.load(directory+"pred_mask_"+str(max_seq_length)+".npy")
    return pred_tokens, pred_X, pred_mask

def load_saved_model(model_path):
    l_bert, model_ckpt = fetch_bert_layer()
    model = load_model(model_path,
                       custom_objects={"BertModelLayer":l_bert,
                                       "argument_candidate_acc":class_acc(3)})
    return model

def pred_model_UNSC(direct_model,max_seq_length=512,
                    direct_save="./data/UNSC/pred/"):
    _,pred_X,_,_ = read_or_create_data_UNSC(max_seq_length)
    model = load_saved_model(direct_model)
    y_pred = model.predict(pred_X,batch_size=128)
    y_pred = np.argmax(y_pred,axis=-1)
    np.save(direct_save+"pred_Yhat_"+str(max_seq_length)+".npy",y_pred)
    return y_pred

def retokenize_UNSC(y_pred,max_seq_length=512):
    (pred_tokens,_,pred_mask,
    speech_ids) = read_or_create_data_UNSC(max_seq_length)
    # TODO remove all special tokens, keep only important tokens
    # TODO modify command-line interface to allow for processing and retokenizing
    # project thoes back to actual tokens along with corresponding labels
    # write to json, perhaps make some automatic pipeline to choose best cases
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="maximum sequence length of tokenized id's")
    required = parser.add_argument_group("required name arguments")
    required.add_argument("--model-dir", required=True, type=str,
                          help="path to model *h5 file")
    args = parser.parse_args()
    if os.path.isfile("./data/UNSC/pred/pred_Yhat_"+
                      str(args.max_seq_length)+".npy"):
        y_pred = np.load("./data/UNSC/pred/pred_Yhat_"+
                      str(args.max_seq_length)+".npy")
    else:
        y_pred = pred_model_UNSC(direct_model=args.model_dir,
                                 max_seq_length=args.max_seq_length)
