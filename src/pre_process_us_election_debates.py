#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import bert
import json
import nltk
import codecs
import argparse
from glob import glob
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from utils.arg_metav_formatter import *
from utils.data_utils import *

def read_us_election_corpus():
    """
    Function to read US election debate text and annotations. Conducts sanity
    check on text vs. annotations to ensure data makes sense.

    Returns:
        raw (list): raw text in corpus
        ann (list): annotations in corpus
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

def flatten(char_sequences,indices=False):
    """
    Function to split and flatten character sequences

    Args:
        char_sequences (list): character sequences
        indices (bool): True to return a list of indices

    Returns:
        flat (list): flattened arguments character sequences
        indices (list): optional list of indices
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
        flat_text (list): character sequences for raw rext from "flatten"
        flat_ann (list): character sequences for tagged text from "flatten"
        spaces (bool): True if spaces are to be annotated, False if not

    Returns:
        flat_text (list): corrected version of flat_text
        flat_ann (list): corrected version of flat_ann
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
        flat_text (list): character sequences for raw rext from "flatten"
        flat_ann (list): character sequences for tagged text from "flatten"
        Tokenizer (object): tokenizer class for either nltk or bert

    Returns:
        split_combined_pruned (list): pruned list of tokens and associated annotations
    """
    # first loop to zip results and check for equal length initial tokens
    split_combined = []
    if "nltk" in str(Tokenizer):
        for i in range(len(flat_text)):
            split_text = [Tokenizer.word_tokenize(el)
                          for el in flat_text[i].split(" ")]
            split_ann = [(el,Counter(el)) for el in flat_ann[i].split("S")]
            assert len(split_ann) == len(split_text)
            split_combined.append(list(zip(split_text,split_ann)))
    elif "bert" in str(Tokenizer):
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
    Function to write parsed text to CoNLL file format

    Args:
        data (list): output from corpus2char/corpus2tokens
        directory (str): directory to store files
        name (str): name of file
        header (str): header of CoNLL stored files
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

def corpus2tokens(tokenizer="bert",max_seq_length=128):
    """
    Function to convert US election corpus to token representation
    and save to pickle

    Args:
        tokenizer (str): whether to use nltk or bert for tokenization
        max_seq_length (int): maximum token sequence length for model
    """
    corpus = read_us_election_corpus()
    tagged = char_tag(corpus,spaces=True)
    flat_text = flatten(corpus[0])
    flat_ann = flatten(tagged)
    assert len(flat_text) == len(flat_ann)
    flat_text,flat_ann = correct_periods(flat_text,flat_ann,spaces=True)
    collection = []
    print("splitting and tokenizing sentences")
    # check for punkt tokenizer
    try:
        nltk.word_tokenize("testing.")
    except LookupError:
        nltk.download('punkt')
    # define albert tokenizer
    if tokenizer == "bert":
        Tokenizer = initialize_bert_tokenizer()
    elif tokenizer == "nltk":
        Tokenizer = nltk.tokenize
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
    return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--no-spaces", action="store_true", default=False,
                        help="if True, spaces will be annotated as arguments"+
                        " in overall span. If False, spaces will be tagged"+
                        " as 'S'.")
    args = parser.parse_args()
    corpus2char(spaces=(not args.no_spaces))
