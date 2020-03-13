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
                             directory="./data/UNSC/eval/"):
    check = glob(directory+"*"+str(max_seq_length)+"*")
    if len(check) < 3:
        (eval_X, eval_mask,
         rel_ids)= corpus2tokenids_UNSC(max_seq_length=max_seq_length)
    else:
        eval_X = np.load(directory+"eval_X_"+str(max_seq_length)+".npy")
        eval_mask = np.load(directory+"eval_mask_"+str(max_seq_length)+".npy")
        with open(directory+"speech_ids_"+str(max_seq_length)+".json","r") as f:
            speech_ids = json.load(f)
    return eval_X, eval_mask, speech_ids

def load_saved_model(model_path):
    l_bert, model_ckpt = fetch_bert_layer()
    model = load_model(model_path,
                       custom_objects={"BertModelLayer":l_bert,
                                       "argument_candidate_acc":class_acc(3)})
    return model

def eval_model_UNSC(direct_model,max_seq_length=512,
                    direct_save="./data/UNSC/eval/"):
    eval_X,_,_ = read_or_create_data_UNSC(max_seq_length)
    model = load_saved_model(direct_model)
    y_pred = model.predict(eval_X,batch_size=128)
    y_pred = np.argmax(y_pred,axis=-1)
    np.save(direct_save+"eval_Yhat_"+str(max_seq_length)+".npy",y_pred)
    return y_pred

def retokenize_UNSC(y_pred,max_seq_length=512):
    _, eval_mask, speech_ids = read_or_create_data_UNSC(max_seq_length)
    # TODO remove all special tokens, keep only important tokens
    # TODO modify command-line interface to allow for processing and retokenizing
    # project thoes back to actual tokens along with corresponding labels
    # Tokenizer.convert_ids_to_tokens([list])
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
    if os.path.isfile("./data/UNSC/eval/eval_Yhat_"+
                      str(args.max_seq_length)+".npy"):
        y_pred = np.load("./data/UNSC/eval/eval_Yhat_"+
                      str(args.max_seq_length)+".npy")
    else:
        y_pred = eval_model_UNSC(direct_model=args.model_dir,
                                 max_seq_length=args.max_seq_length)
