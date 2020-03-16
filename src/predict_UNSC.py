#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from tensorflow.keras.models import load_model
from train_USElectionDebates import read_or_create_data_US
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
                    direct_save="./data/UNSC/pred/",
                    force_pred=False):
    if not force_pred and os.path.isfile("./data/UNSC/pred/pred_Yhat_"+
                      str(args.max_seq_length)+".npy"):
        y_pred = np.load("./data/UNSC/pred/pred_Yhat_"+
                      str(args.max_seq_length)+".npy")
    else:
        _,pred_X,_ = read_or_create_data_UNSC(max_seq_length)
        model = load_saved_model(direct_model)
        y_pred = model.predict(pred_X,batch_size=128)
        y_pred = np.argmax(y_pred,axis=-1)
        np.save(direct_save+"pred_Yhat_"+str(max_seq_length)+".npy",y_pred)
    return y_pred

def simplify_results(y_pred,max_seq_length=512,
                     directory="./data/UNSC/pred/"):
    pred_tokens,_,pred_mask = read_or_create_data_UNSC(max_seq_length)
    _,_,_,_,label_map = read_or_create_data_US(max_seq_length)
    label_map_inverse = {item[1]:item[0] for item in label_map.items()}
    keys = list(pred_tokens.keys())
    clean_results = {}
    for i in tqdm(range(pred_mask.shape[0])):
        clean_results[keys[i]]=[(pred_tokens[keys[i]][j],
                                        label_map_inverse[y_pred[i,j]])
         for j,binary in enumerate(pred_mask[i].tolist()) if binary == 1]
    with open(directory+"pred_clean_"+str(max_seq_length)+".json","w") as f:
        json.dump(clean_results,f)
    return clean_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="maximum sequence length of tokenized id's")
    parser.add_argument("--force-pred", action="store_true", default=False,
                        help="option to force redoing prediction despite"+
                        " presence of already produced binary")
    required = parser.add_argument_group("required name arguments")
    required.add_argument("--model-dir", required=True, type=str,
                          help="path to model *h5 file")
    args = parser.parse_args()
    print("Loading model predictions")
    y_pred = pred_model_UNSC(direct_model = args.model_dir,
                             max_seq_length = args.max_seq_length,
                             force_pred = args.force_pred)
    print("Simplifying model predictions for human readability")
    clean_results = simplify_results(y_pred,args.max_seq_length)
