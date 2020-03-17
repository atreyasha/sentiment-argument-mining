#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import json
import bert
import nltk
import pyreadr
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from pre_process_USElectionDebates import initialize_bert_tokenizer
from utils.arg_metav_formatter import *

def load_UNSC():
    """
    Function to load raw UNSC RData

    Returns:
        ids (list[str]): list of unique IDs for speeches
        flat_text (list[str]): list of raw speech text
    """
    data = pyreadr.read_r("./data/UNSC/docs.RData")["raw_docs"]
    ids = data["doc_id"].tolist()
    flat_text = data["text"].tolist()
    return ids, flat_text

def basic_text_cleaning(flat_text):
    """
    Function to clean raw UNSC data; removes odd spacings and artefacts

    Args:
        flat_text (list[str]): raw flattened UNSC text from
        load_UNSC function

    Returns:
        flat_text (list[str]): raw cleaned UNSC text
    """
    for i, text in enumerate(tqdm(flat_text)):
        span = re.search(r"^.*:(\s)?",text)
        if span is None:
            pass
        else:
            span = span.span()
            if span[0] == 0:
                flat_text[i] = flat_text[i][span[1]:]
        flat_text[i] = re.sub("(\n)(\s+)?(\n+)?"," ",flat_text[i])
        flat_text[i] = re.sub(r"\([^)]*\)","",flat_text[i])
    return flat_text

def project_to_ids_UNSC(Tokenizer,data,max_seq_length=512):
    """
    Function to map data to indices in the albert vocabulary, as well as
    adding special bert tokens such as [CLS] and [SEP]

    Args:
        Tokenizer (bert.tokenization.albert_tokenization.FullTokenizer):
        tokenizer class for bert tokenizer
        data (list): input data containing albert tokens
        max_seq_length (int): maximum sequence length to be used in training

    Returns:
        (list[str]): input albert tokens including special tokens
        (np.ndarray): input albert token IDs
        (np.ndarray): input mask indicating which token is relevant to outcome,
        this includes all corpus tokens and excludes all bert special tokens
    """
    input_tokens = []
    input_ids = []
    input_mask = []
    for instance_set in tqdm(data):
        input_ids_sub = ["[CLS]"]
        input_mask_sub = [0]
        for i in range(len(instance_set[1])):
            input_ids_sub.extend(instance_set[1][i])
            input_mask_sub.extend([1]*len(instance_set[1][i]))
        input_ids_sub.extend(["[SEP]"])
        input_mask_sub.extend([0])
        assert (len(input_ids_sub) == len(input_mask_sub))
        input_ids_sub.extend(["<pad>"]*(max_seq_length-len(input_ids_sub)))
        input_mask_sub.extend([0]*(max_seq_length-len(input_mask_sub)))
        assert (len(input_ids_sub) == len(input_mask_sub) == max_seq_length)
        input_tokens.append(input_ids_sub.copy())
        input_ids_sub = Tokenizer.convert_tokens_to_ids(input_ids_sub)
        input_ids.append(input_ids_sub)
        input_mask.append(input_mask_sub)
    return input_tokens, np.array(input_ids), np.array(input_mask)

def summary_info_UNSC(collection,ids,
                      directory="./data/UNSC/pred/"):
    """
    Function to write summary statistics on token types to file

    Args:
        collection (list): data containing token and types
        indices (int): mapping of `collection` to debate segments
        directory (str): directory to output files
    """
    # get respective token counts
    lens = [len(nltk.tokenize.word_tokenize(el)) for el in tqdm(collection)]
    # write to csv file
    with open(directory+"stats_tokens.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["id","len"])
        writer.writerows(list(zip(ids,lens)))
    return lens

def corpus2tokenids_UNSC(max_seq_length=512,
                         directory="./data/UNSC/pred/"):
    """
    Aggregate function to produce bert-operational prediction data from
    the UNSC corpus

    Args:
        max_seq_length (int): maximum sequence length to be used in training
        directory (str): directory to output files

    Returns:
        pred_tokens (dict): mapping between unique UNSC speech IDs and
        tokenized input
        pred_X (np.ndarray): input albert token IDs
        pred_mask (np.ndarray): input mask indicating which token is relevant
        to outcome, this includes all corpus tokens and excludes
        all bert special tokens
    """
    print("Loading UNSC data from RData format")
    ids, flat_text = load_UNSC()
    print("Performing basic cleaning of data")
    flat_text = basic_text_cleaning(flat_text)
    try:
        nltk.tokenize.sent_tokenize("testing. testing")
    except LookupError:
        nltk.download('punkt')
    # get summary information on corpus
    print("Computing corpus summary statistics")
    lens = summary_info_UNSC(flat_text,ids)
    # intialize variables
    collection = []
    Tokenizer = initialize_bert_tokenizer()
    preprocess = bert.albert_tokenization.preprocess_text
    # enter main tokenization loop
    print("Tokenizing corpus")
    for i in tqdm(range(len(flat_text))):
        if lens[i] > max_seq_length:
            continue
        else:
            tmp_1 = []
            sents = nltk.tokenize.sent_tokenize(flat_text[i])
            for sent in sents:
                tmp_2 = []
                for token in sent.split(" "):
                    tokenized = Tokenizer.tokenize(preprocess(token,lower=True))
                    for tokenized_token in tokenized:
                        tmp_2.append(tokenized_token)
                tmp_1.append(tmp_2)
            collection.append([i,tmp_1])
    # check data length and remove long sentences
    to_remove = []
    print("Removing corpus elements which are too long")
    for i,sent_set in enumerate(collection):
        token_count = sum([1 for sent in sent_set[1] for token in sent])
        length = token_count+2
        if length > max_seq_length:
            to_remove.append(i)
    collection = [sent_set for i,sent_set in enumerate(collection)
                  if i not in to_remove]
    print("Projecting text to indices")
    pred_tokens, pred_X, pred_mask = project_to_ids_UNSC(Tokenizer,collection,
                                                    max_seq_length)
    ids_tokens = [ids[sub[0]] for sub in collection]
    pred_tokens = {ids_tokens[i]:[token.decode("utf-8") if type(token) is bytes
                    else token for token in sent_set]
                   for i, sent_set in enumerate(pred_tokens)}
    np.save(directory+"pred_X_"+str(max_seq_length)+".npy",pred_X)
    np.save(directory+"pred_mask_"+str(max_seq_length)+".npy",pred_mask)
    with open(directory+"pred_tokens_"+str(max_seq_length)+".json","w") as f:
        json.dump(pred_tokens,f,ensure_ascii=False)
    return pred_tokens, pred_X, pred_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="maximum sequence length of tokenized id's")
    args = parser.parse_args()
    corpus2tokenids_UNSC(max_seq_length=args.max_seq_length)
