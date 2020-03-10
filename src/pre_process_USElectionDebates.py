#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import bert
import json
import nltk
import codecs
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from utils.arg_metav_formatter import *

def read_us_election_corpus():
    """
    Function to read US election debate text and annotations. Conducts sanity
    check on text vs. annotations to ensure data makes sense.

    Returns:
        raw (list[str]): raw text in corpus
        ann (list[str]): annotations in corpus
    """
    # read raw text into memory by sorting first through indices
    files_raw = glob("./data/USElectionDebates/*.txt")
    indices = [int(re.sub(r".*\/(\d+)_(\d+).*","\g<1>",File))
               for File in files_raw]
    files_raw = [f for i,f in sorted(zip(indices,files_raw))]
    files_ann = glob("./data/USElectionDebates/*.ann")
    indices = [int(re.sub(r".*\/(\d+)_(\d+).*","\g<1>",File))
               for File in files_ann]
    files_ann = [f for i,f in sorted(zip(indices,files_ann))]
    # read raw text into pythonic list
    raw = []
    ann = []
    print("reading raw corpus files")
    for File in tqdm(files_raw):
        with codecs.open(File,"r",encoding="utf-8") as f:
            raw.append(f.read())
    # read annotations and append to merged list
    print("reading corpus annotations")
    for i,File in tqdm(enumerate(files_ann),total=len(files_ann)):
        with codecs.open(File,"r",encoding="utf-8") as f:
            ann_data = f.readlines()
        for annotation in ann_data:
            hold = annotation.replace("\n","").split("\t")
            split = re.split(r"\s+|\;",hold[1])
            spans = [int(el) for el in split[1:]]
            spans = list(zip(*(iter(spans),)*2))
            # check that individual subsets match overall string
            assert " ".join(raw[i][pair[0]:pair[1]]
                            for pair in spans) == hold[2]
            ann.append([i,hold[0],split[0],spans,hold[2]])
    return raw,ann

def char_tag(corpus,spaces=False):
    """
    Function to convert raw US election debate text
    and annotations into tagged character sequence

    Args:
        corpus (list[str],list[str]): output of "read_us_election_corpus"
        spaces (bool): True if spaces should be marked, False if
        they should be marked same as span

    Returns:
        (list[str]): each component contains annotated characters
    """
    # create empty list
    tagged = []
    # tag basic cases
    print("tagging void argument cases")
    for i,raw in enumerate(tqdm(corpus[0])):
        char_raw = list(raw)
        for j,char in enumerate(char_raw):
            if spaces:
                if char not in [" ","\n","\r"]:
                    char_raw[j] = "N"
                elif char == " ":
                    char_raw[j] = "S"
            else:
                if char not in ["\n","\r"]:
                    char_raw[j] = "N"
        tagged.append(char_raw)
    # tag claims and premises
    print("tagging claims and premises")
    for i in tqdm(range(len(tagged))):
        for annotation in corpus[1]:
            if annotation[0] == i:
                ann_type = annotation[2]
                spans = annotation[3]
                for z,pair in enumerate(spans):
                    for j in range(pair[0],pair[1]):
                        char = tagged[i][j]
                        if spaces:
                            if char not in ["S","\n","\r"]:
                                if ann_type == "Claim":
                                    tagged[i][j] = "C"
                                elif ann_type == "Premise":
                                    tagged[i][j] = "P"
                        else:
                            if char not in ["\n","\r"]:
                                if ann_type == "Claim":
                                    tagged[i][j] = "C"
                                elif ann_type == "Premise":
                                    tagged[i][j] = "P"
    # join and return final tag
    print("returning final object")
    return ["".join(char for char in segment) for segment in tqdm(tagged)]

def flatten(char_sequences,indices=False):
    """
    Function to split and flatten character sequences

    Args:
        char_sequences (list[str]): character sequences
        indices (bool): True to return a list of indices

    Returns:
        flat (list[str]): flattened arguments character sequences
        indices (list[int]): optional list of indices
    """
    flat = []
    if indices:
        indices = []
        for i, speech in enumerate(char_sequences):
            split = re.split(r"\r\n|\n|\r|\n\r",speech)
            for j, segment in enumerate(split):
                if segment != "":
                    flat.append(segment)
                    indices.append([i,j])
        # return both lists
        return indices, flat
    else:
        for speech in char_sequences:
            split = re.split(r"\r\n|\n|\r|\n\r",speech)
            for segment in split:
                if segment != "":
                    flat.append(segment)
        return flat

def correct_periods(flat_text,flat_ann,spaces=False):
    """
    Function to add spaces where incorrect periods are present

    Args:
        flat_text (list[str]): character sequences for raw rext from "flatten"
        flat_ann (list[str]): character sequences for tagged text from "flatten"
        spaces (bool): True if spaces are to be annotated, False if not

    Returns:
        flat_text (list[str]): corrected version of flat_text
        flat_ann (list[str]): corrected version of flat_ann
    """
    for i in range(len(flat_text)):
        run = True
        while run:
            match = re.search(r"[a-z]\.[A-Z]",flat_text[i])
            if match == None:
                run = False
            else:
                flat_text[i] = (flat_text[i][:match.end()-1] +
                                " " + flat_text[i][match.end()-1:])
                if spaces:
                    flat_ann[i] = (flat_ann[i][:match.end()-1] +
                                "S" + flat_ann[i][match.end()-1:])
                else:
                    forward_ann = flat_ann[i][match.end()-1]
                    flat_ann[i] = (flat_ann[i][:match.end()-1] +
                                forward_ann + flat_ann[i][match.end()-1:])
    return flat_text,flat_ann

def tokenize(flat_text,flat_ann,Tokenizer):
    """
    Function to prune and tokenize corpus

    Args:
        flat_text (list[str]): character sequences for raw rext from "flatten"
        flat_ann (list[str]): character sequences for tagged text from "flatten"
        Tokenizer (bert.tokenization.albert_tokenization.FullTokenizer):
        tokenizer class for bert tokenizer

    Returns:
        split_combined_pruned (list[str]): pruned list of tokens and associated
        annotations
    """
    # first loop to zip results and check for equal length initial tokens
    split_combined = []
    preprocess = bert.albert_tokenization.preprocess_text
    for i in range(len(flat_text)):
        split_text = [Tokenizer.tokenize(preprocess(el,lower=True))
                    for el in flat_text[i].split(" ")]
        split_ann = [(el,Counter(el)) for el in flat_ann[i].split("S")]
        assert len(split_ann) == len(split_text)
        split_combined.append(list(zip(split_text,split_ann)))
    # prune tokens to ensure proper splits
    split_combined_pruned = []
    # next loop to check and combine results
    for segment in split_combined:
        temp = []
        for token_set in segment:
            if len(token_set[0]) > 0:
                most_common = token_set[1][1].most_common()[0][0]
                if (sum([len(token) for token in token_set[0]]) ==
                    len(token_set[1][0])):
                    if len(token_set[1][1]) == 1:
                        for token in token_set[0]:
                            temp.append([token,most_common])
                    else:
                        count = 0
                        for token in token_set[0]:
                            tag = Counter(token_set[1][0]
                                        [count:(count+len(token))])
                            most_common_local = tag.most_common()[0][0]
                            temp.append([token,most_common_local])
                            count += len(token)
                else:
                    for token in token_set[0]:
                        temp.append([token,most_common])
        split_combined_pruned.append(temp)
    return split_combined_pruned

def post_process(data):
    """
    Function to unzip tokenized data for output

    Args:
        data (list): zipped tokenized data

    Returns:
        token_list (list): unzipped tokenized data
    """
    token_list = []
    for i, sent_set in enumerate(data):
        tmp_sent = []
        tmp_ann = []
        for j,sent in enumerate(sent_set[1]):
            unzipped = list(zip(*sent))
            tmp_sent.append(list(unzipped[0]))
            tmp_ann.append(list(unzipped[1]))
        token_list.append([data[i][0],tmp_sent,tmp_ann])
    return token_list

def write_to_json(data,directory,name):
    """
    Function to write parsed text to JSON file

    Args:
        data (list): output from corpus2char
        directory (str): directory to store files
        name (str): name of file
    """
    char_dict = {}
    for i,j,text,ann in data:
        if i not in char_dict.keys():
            char_dict[i] = {}
        char_dict[i][j] = {"text":text,
                        "ann":ann}
    with open(directory+name,"w") as f:
        json.dump(char_dict,f)

def corpus2char(directory="./data/USElectionDebates/corpus/",
                spaces=True):
    """
    Function to convert US election corpus to character representation
    and save to json

    Args:
        directory (str): base file directory on which to store output
        spaces (bool): True to tag spaces as separate character
    """
    if not directory.endswith("/"):
        directory += "/"
    corpus = read_us_election_corpus()
    tagged = char_tag(corpus,spaces=spaces)
    indices,flat_text = flatten(corpus[0],indices=True)
    flat_ann = flatten(tagged)
    assert len(flat_text) == len(flat_ann)
    flat_text,flat_ann = correct_periods(flat_text,flat_ann,spaces=spaces)
    corpus = [[indices[i][0],indices[i][1],text,flat_ann[i]]
              for i,text in enumerate(flat_text)]
    write_to_json(corpus,directory,"corpus.json")

def initialize_bert_tokenizer():
    """
    Function to initialize the bert tokenizer

    Returns:
        Tokenizer (bert.tokenization.albert_tokenization.FullTokenizer)
    """
    model_name = "albert_base_v2"
    model_dir = bert.fetch_google_albert_model(model_name, ".models")
    spm = os.path.join(model_dir,"30k-clean.model")
    vocab = os.path.join(model_dir,"30k-clean.vocab")
    Tokenizer = bert.albert_tokenization.FullTokenizer(vocab,
                                                       spm_model_file=spm)
    return Tokenizer

def project_to_ids(Tokenizer,train_data,label_id_map,max_seq_length=512):
    """
    Function to map data to indices in the albert vocabulary, as well as
    adding special bert tokens such as [CLS] and [SEP]

    Args:
        Tokenizer (bert.tokenization.albert_tokenization.FullTokenizer):
        tokenizer class for bert tokenizer
        train_data (list): input data containing albert tokens and label types
        label_id_map (dict): dictionary mapping label type to integer
        max_seq_length (int): maximum sequence length to be used in training

    Returns:
        (np.ndarray): input albert IDs
        (np.ndarray): output label IDs
        (np.ndarray): output mask indicating which token is relevant to outcome,
        this includes all corpus tokens and excludes all bert special tokens
    """
    input_ids = []
    label_ids = []
    output_mask = []
    print("projecting text to indices")
    for instance_set in tqdm(train_data):
        input_ids_sub = ["[CLS]"]
        label_ids_sub = ["[CLS]"]
        output_mask_sub = [0]
        for i in range(len(instance_set[1])):
            input_ids_sub.extend(instance_set[1][i])
            label_ids_sub.extend(instance_set[2][i])
            output_mask_sub.extend([1]*len(instance_set[1][i]))
            input_ids_sub.extend(["[SEP]"])
            label_ids_sub.extend(["[SEP]"])
            output_mask_sub.extend([0])
        assert (len(input_ids_sub) == len(label_ids_sub)
                == len(output_mask_sub))
        input_ids_sub.extend(["<pad>"]*(max_seq_length-len(input_ids_sub)))
        label_ids_sub.extend(["<pad>"]*(max_seq_length-len(label_ids_sub)))
        output_mask_sub.extend([0]*(max_seq_length-
                                    len(output_mask_sub)))
        assert (len(input_ids_sub) == len(label_ids_sub)
                == len(output_mask_sub) == max_seq_length)
        input_ids_sub = Tokenizer.convert_tokens_to_ids(input_ids_sub)
        label_ids_sub = [label_id_map[label] for label in label_ids_sub]
        input_ids.append(input_ids_sub)
        label_ids.append(label_ids_sub)
        output_mask.append(output_mask_sub)
    return np.array(input_ids), np.array(label_ids), np.array(output_mask)

def summary_info(collection,indices,
                 directory="./data/USElectionDebates/corpus/"):
    """
    Function to write summary statistics on token types to file

    Args:
        collection (list): data containing token and types
        indices (int): mapping of `collection` to debate segments
        directory (str): directory to output files
    """
    new_collection = []
    # get respective token counts
    for i,el in enumerate(collection):
        new_collection.append([indices[i][0]])
        tmp = []
        for sub_el in el:
            for sub_sub_el in sub_el:
                tmp.append(sub_sub_el[1])
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
    with open(directory+"stats_tokens.csv","w") as f:
        writer = csv.writer(f)
        writer.writerow(["debate","N","C","P"])
        writer.writerows(new_collection)

def corpus2tokenids(max_seq_length=512,
                    directory="./data/USElectionDebates/training/"):
    """
    Aggregate function to produce bert-operational training data from
    US-Election-Debate corpus

    Args:
        max_seq_length (int): maximum sequence length to be used in training
        directory (str): directory to output files

    Returns:
        train_X (np.ndarray): training data IDs
        train_Y (np.ndarray): training data labels
        test_X (np.ndarray): testing data IDs
        test_Y (np.ndarray): testing data labels
        label_map (dict): mapping from label to integer ID for labels
    """
    label_map = {"<pad>":0,"[CLS]":1,"[SEP]":2,"N":3,"C":4,"P":5}
    corpus = read_us_election_corpus()
    tagged = char_tag(corpus,spaces=True)
    indices,flat_text = flatten(corpus[0],indices=True)
    flat_ann = flatten(tagged)
    assert len(flat_text) == len(flat_ann)
    flat_text,flat_ann = correct_periods(flat_text,flat_ann,spaces=True)
    # remove initial capitalized NER reference, might assist with debiasing
    for i, text in enumerate(flat_text):
        span = re.search("^[A-Z]*\\:\\s",text)
        if span is None:
            continue
        else:
            span = span.span()
            if span[0] == 0:
                flat_text[i] = flat_text[i][span[1]:]
                flat_ann[i] = flat_ann[i][span[1]:]
                assert len(flat_text[i]) == len(flat_ann[i])
    collection = []
    print("splitting and tokenizing sentences")
    Tokenizer = initialize_bert_tokenizer()
    # ensure nltk sentence tokenizer is present
    try:
        nltk.tokenize.sent_tokenize("testing. testing")
    except LookupError:
        nltk.download('punkt')
    # enter main loop
    for i in tqdm(range(len(flat_text))):
        sub_text = nltk.tokenize.sent_tokenize(flat_text[i])
        sub_ann = []
        for j,chunk in enumerate(sub_text):
            span = re.search(re.escape(chunk),flat_text[i]).span()
            sub_ann.append(flat_ann[i][span[0]:span[1]])
            assert len(sub_ann[j]) == len(chunk)
            flat_text[i] = flat_text[i][span[1]:]
            flat_ann[i] = flat_ann[i][span[1]:]
        collection.append(tokenize(sub_text,sub_ann,Tokenizer))
    # print out summary info for later visualization
    summary_info(collection,indices)
    # check data length and remove long sentences
    to_remove = []
    for i,sent_set in enumerate(collection):
        token_count = sum([1 for sent in sent_set for token in sent])
        length = token_count+len(sent_set)+1
        if length > max_seq_length:
            to_remove.append(i)
    collection = [sent_set for i,sent_set in enumerate(collection)
                  if i not in to_remove]
    collection = [[i,sent_set] for i,sent_set in enumerate(collection)]
    # split data into train and test sets
    train, test = train_test_split(collection,test_size=0.33,
                                   random_state=42)
    train = post_process(train)
    test = post_process(test)
    train_X, train_Y, _ = project_to_ids(Tokenizer,train,label_map,
                                         max_seq_length)
    test_X, test_Y, _ = project_to_ids(Tokenizer,test,label_map,
                                       max_seq_length)
    np.save(directory+"train_X_"+str(max_seq_length)+".npy",train_X)
    np.save(directory+"train_Y_"+str(max_seq_length)+".npy",train_Y)
    np.save(directory+"test_X_"+str(max_seq_length)+".npy",test_X)
    np.save(directory+"test_Y_"+str(max_seq_length)+".npy",test_Y)
    # write json label map
    with open(directory+"label_map.json","w") as f:
        json.dump(label_map,f)
    return train_X, train_Y, test_X, test_Y, label_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="maximum sequence length of tokenized id's")
    args = parser.parse_args()
    corpus2char()
    corpus2tokenids(max_seq_length=args.max_seq_length)
