#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
import json
import nltk
import codecs
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from collections import Counter
from obj.arg_metav_formatter import *

def read_us_election_corpus():
    """
    Function to read US election debate text and annotations. Conducts sanity
    check on text vs. annotations to ensure data makes sense.

    Returns:
        (list,list): raw text and annotated argument candidates respectively
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
        corpus (list,list): output of "read_us_election_corpus"
        spaces (bool): True if spaces should be marked, False if
        they should be marked same as span

    Returns:
        (list): each component contains annotated characters
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

def flatten(char_sequences):
    """
    Function to split and flatten character sequences

    Args:
        char_sequences (list): character sequences

    Returns:
        (list): flattened arguments character sequences
    """
    flat = []
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
        flat_text (list): character sequences for raw rext from "flatten"
        flat_ann (list): character sequences for tagged text from "flatten"
        spaces (bool): True if spaces are to be annotated, False if not

    Returns:
        (list,list): corrected versions of flat_text and flat_ann
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

def tokenize(flat_text,flat_ann):
    """
    Function to prune and tokenize corpus

    Args:
        flat_text (list): character sequences for raw rext from "flatten"
        flat_ann (list): character sequences for tagged text from "flatten"

    Returns:
        (list): pruned list of tokens and associated annotations
    """
    # first loop to zip results and check for equal length initial tokens
    split_combined = []
    print("tokenizing and performing sanity checks")
    try:
        nltk.word_tokenize("testing.")
    except LookupError:
        nltk.download('punkt')
    for i in tqdm(range(len(flat_text))):
        split_text = [nltk.word_tokenize(el) for el in flat_text[i].split(" ")]
        split_ann = [(el,Counter(el)) for el in flat_ann[i].split("S")]
        assert len(split_ann) == len(split_text)
        split_combined.append(list(zip(split_text,split_ann)))
    # prune tokens to ensure proper splits
    split_combined_pruned = []
    # next loop to check and combine results
    print("tagging tokens")
    for segment in tqdm(split_combined):
        temp = []
        for token_set in segment:
            if len(token_set[0]) > 0:
                most_common = token_set[1][1].most_common()[0][0]
                if len("".join(token_set[0])) == len(token_set[1][0]):
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

def roundup(x,nearest=100):
    """
    Function to round integer to nearest integer (eg. 100)

    Args:
        x (float or int): number which should be rounded up
        nearest (int): nearest integer on which to round up

    Returns:
        (int): rounded up integer
    """
    return int(np.ceil(x/nearest))*int(nearest)

def build_token_dict(token_list):
    """
    Function to round integer to nearest integer (eg. 100)

    Args:
        token_list (list): list containing list of tokenized segments

    Returns:
        (dict): dictionary and indices of all tokens
    """
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2
    }
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict

def text2encoding(source_tokens,target_tokens):
    """
    Function to convert token segments into indices for training

    Args:
        source_tokens (list): source or input tokens
        target_tokens (list): target or ouput tokens

    Returns:
        encode_input (numpy.ndarray): encoded input tokens
        decode_input (numpy.ndarray): decoded input tokens
        decode_output (numpy.ndarray): decoded output tokens
        source_token_dict (dict): source tokens and indices
        target_token_dict (dict): target tokens and indices
    """
    print("building dictionaries")
    source_token_dict = build_token_dict(source_tokens)
    target_token_dict = build_token_dict(target_tokens)
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}
    # add special tokens
    print("adding special tokens")
    encode_tokens = [['<START>'] + tokens + ['<END>']
                     for tokens in source_tokens]
    decode_tokens = [['<START>'] + tokens + ['<END>']
                     for tokens in target_tokens]
    output_tokens = [tokens + ['<END>', '<PAD>']
                     for tokens in target_tokens]
    # padding
    print("padding tokens")
    source_max_len = roundup(max(map(len, encode_tokens)))
    target_max_len = roundup(max(map(len, decode_tokens)))
    encode_tokens = [tokens + ['<PAD>'] *
                     (source_max_len - len(tokens))
                     for tokens in tqdm(encode_tokens)]
    decode_tokens = [tokens + ['<PAD>'] *
                    (target_max_len - len(tokens))
                     for tokens in tqdm(decode_tokens)]
    output_tokens = [tokens + ['<PAD>'] *
                    (target_max_len - len(tokens))
                     for tokens in tqdm(output_tokens)]
    # map to indices
    print("mapping tokens to indices")
    encode_input = np.array([list(map(lambda x: source_token_dict[x], tokens))
                    for tokens in tqdm(encode_tokens)])
    decode_input = np.array([list(map(lambda x: target_token_dict[x], tokens))
                    for tokens in tqdm(decode_tokens)])
    decode_output = np.array([list(map(lambda x: [target_token_dict[x]],
                                       tokens))
                              for tokens in tqdm(output_tokens)])
    return (encode_input,decode_input,decode_output,
            source_token_dict,target_token_dict)

def corpus2char(file_path="./data/pre-processed/"):
    """
    Function to convert US election corpus to numpy character encodings

    Args:
        file_path (str): base file directory on which to store output
    """
    corpus = read_us_election_corpus()
    tagged = char_tag(corpus,spaces=False)
    flat_text = flatten(corpus[0])
    flat_ann = flatten(tagged)
    assert len(flat_text) == len(flat_ann)
    flat_text,flat_ann = correct_periods(flat_text,flat_ann,spaces=False)
    with open(file_path+"characterized_corpus.tsv","w") as f:
        writer = csv.writer(f,delimiter="\t")
        header = ["i","char_sequence->[P=premise,C=claim,N=None]"]
        writer.writerow(header)
        for i in range(len(flat_text)):
            writer.writerow([i,flat_text[i]])
            writer.writerow([i,flat_ann[i]])
    source_char = [list(text) for text in flat_text]
    target_char = [list(ann) for ann in flat_ann]
    (encode_input,decode_input,decode_output,
            source_char_dict,target_char_dict) = text2encoding(source_char,
                                                                 target_char)
    np.save(file_path+"characterized_encode_input.npy",encode_input)
    np.save(file_path+"characterized_decode_input.npy",decode_input)
    np.save(file_path+"characterized_decode_output.npy",
            decode_output)
    with open(file_path+"characterized_source_dict.json","w") as f:
        json.dump(source_char_dict,f)
    with open(file_path+"characterized_target_dict.json","w") as f:
        json.dump(target_char_dict,f)

def corpus2tokens(file_path="./data/pre-processed/"):
    """
    Function to convert US election corpus to numpy token encodings

    Args:
        file_path (str): base file directory on which to store output
    """
    corpus = read_us_election_corpus()
    tagged = char_tag(corpus,spaces=True)
    flat_text = flatten(corpus[0])
    flat_ann = flatten(tagged)
    assert len(flat_text) == len(flat_ann)
    flat_text,flat_ann = correct_periods(flat_text,flat_ann,spaces=True)
    split_combined = tokenize(flat_text,flat_ann)
    with open(file_path+"tokenized_corpus.tsv","w") as f:
        writer = csv.writer(f,delimiter="\t")
        writer.writerow(["i","token","annotation->[P=premise,C=claim,N=None]"])
        for i,token_set in enumerate(split_combined):
            for token,tag in token_set:
                writer.writerow([i,token,tag])
    source_tokens = [[token[0] for token in token_set]
                     for token_set in split_combined]
    target_tokens = [[token[1] for token in token_set]
                     for token_set in split_combined]
    (encode_input,decode_input,decode_output,
            source_token_dict,target_token_dict) = text2encoding(source_tokens,
                                                              target_tokens)
    np.save(file_path+"tokenized_encode_input.npy",encode_input)
    np.save(file_path+"tokenized_decode_input.npy",decode_input)
    np.save(file_path+"tokenized_decode_output.npy",decode_output)
    with open(file_path+"tokenized_source_dict.json","w") as f:
        json.dump(source_token_dict,f)
    with open(file_path+"tokenized_target_dict.json","w") as f:
        json.dump(target_token_dict,f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--dtype", type=str, default="tokens",
                        help="which type of data pre-processing;"+
                        " either 'tokens', 'char' or 'both'")
    args = parser.parse_args()
    assert args.dtype in ["tokens","char","both"]
    if args.dtype == "tokens":
        corpus2tokens()
    elif args.dtype == "char":
        corpus2char()
    elif args.dtype == "both":
        print("processing tokens")
        corpus2tokens()
        print("processing characters")
        corpus2char()
