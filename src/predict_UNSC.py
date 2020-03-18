#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import logging
import logging.config
logging.config.fileConfig('./utils/logging.conf')
import numpy as np
from tqdm import tqdm
from glob import glob
from collections  import Counter
from tensorflow.keras.models import load_model
from train_USElectionDebates import read_or_create_data_US
from pre_process_UNSC import *
from utils.arg_metav_formatter import *
from utils.model_utils import *

def read_or_create_data_UNSC(max_seq_length=512,
                             directory="./data/UNSC/pred/"):
    """
    Function to either read or create UNSC corpus data

    Args:
        max_seq_length (int): maximum sequence length to be used in training
        directory (str): directory to find files

    Returns:
        pred_tokens (dict): mapping between unique UNSC speech IDs and
        tokenized input
        pred_X (np.ndarray): input albert token IDs
        pred_mask (np.ndarray): input mask indicating which token is relevant
        to outcome, this includes all corpus tokens and excludes
        all bert special tokens
    """
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
    """
    Function to load a saved keras model

    Args:
        model_path (str): path to *h5 keras model

    Returns:
        model (tensorflow.python.keras.engine.training.Model): saved keras model
    """
    l_bert, model_ckpt = fetch_bert_layer()
    model = load_model(model_path,
                       custom_objects={"BertModelLayer":l_bert,
                                       "argument_candidate_acc":class_acc(3)})
    return model

def pred_model_UNSC(direct_model,max_seq_length=512,
                    direct_save="./data/UNSC/pred/",
                    force_pred=False):
    """
    Predict given saved model on UNSC corpus

    Args:
        direct_model (str): path to *h5 keras model
        max_seq_length (int): maximum sequence length to be used in training
        direct_save (str): directory where to save predictions
        force_pred (bool): whether to forcefully predict when an cached
        prediction already exists

    Returns:
        y_pred (np.ndarray): model predictions on UNSC corpus
    """
    if not force_pred and os.path.isfile("./data/UNSC/pred/pred_Yhat_"+
                      str(max_seq_length)+".npy"):
        y_pred = np.load("./data/UNSC/pred/pred_Yhat_"+
                      str(max_seq_length)+".npy")
    else:
        _,pred_X,_ = read_or_create_data_UNSC(max_seq_length)
        model = load_saved_model(direct_model)
        y_pred = model.predict(pred_X,batch_size=128)
        y_pred = np.argmax(y_pred,axis=-1)
        np.save(direct_save+"pred_Yhat_"+str(max_seq_length)+".npy",y_pred)
    return y_pred

def summary_info_UNSC_pred(collection,max_seq_length=512,
                           directory="./data/UNSC/pred/"):
    """
    Function to write summary statistics on token types to file

    Args:
        collection (list): data containing token and types
        max_seq_length (int): maximum sequence length to be used in training
        directory (str): directory to output files
    """
    new_collection = []
    # get respective token counts
    for i,el in enumerate(list(collection.keys())):
        new_collection.append([el])
        tmp = []
        for sub_el in collection[el]:
            tmp.append(sub_el[1])
        local_dict = dict(Counter(tmp))
        try:
            N = local_dict["N"]
        except KeyError:
            N = 0
        try:
            C = local_dict["C"]
        except KeyError:
            C = 0
        try:
            P = local_dict["P"]
        except KeyError:
            P = 0
        new_collection[i].extend([N,C,P])
    # write to csv file
    with open(directory+"pred_tokens_stats_"+
              str(max_seq_length)+".csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["speech","N","C","P"])
        writer.writerows(new_collection)

def simplify_results(y_pred,max_seq_length=512,
                     directory="./data/UNSC/pred/"):
    """
    Simplify model predictions on UNSC corpus to human readable format

    Args:
        y_pred (np.ndarray): model predictions on UNSC corpus
        max_seq_length (int): maximum sequence length to be used in training
        directory (str): directory where to save results

    Returns:
        clean_results (dict): simplified dictionary mapping from speech ID's
        to saved model predictions
    """
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
        json.dump(clean_results,f,ensure_ascii=False)
    # execute pipeline to get summary info
    summary_info_UNSC_pred(clean_results)
    return clean_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="maximum sequence length of tokenized id's")
    parser.add_argument("--force-pred", action="store_true", default=False,
                        help="option to force redoing prediction despite"+
                        " presence of already produced binary")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="0 for no text, 1 for verbose text")
    required = parser.add_argument_group("required name arguments")
    required.add_argument("--model-dir", required=True, type=str,
                          help="path to model *h5 file")
    args = parser.parse_args()
    if args.verbosity == 1:
        logger = logging.getLogger('base')
    else:
        logger = logging.getLogger('root')
    logger.info("Loading model predictions, might take some time...")
    y_pred = pred_model_UNSC(direct_model = args.model_dir,
                             max_seq_length = args.max_seq_length,
                             force_pred = args.force_pred)
    logger.info("Simplifying model predictions for human readability")
    clean_results = simplify_results(y_pred,args.max_seq_length)
