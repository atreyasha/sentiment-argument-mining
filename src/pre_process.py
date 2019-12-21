#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import copy
import nltk
import codecs
from glob import glob
from tqdm import tqdm

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
    for File in tqdm(files_raw):
        with codecs.open(File,"r",encoding="utf-8") as f:
            raw.append(f.read())
    # read annotations and append to merged list
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

def char_tag(corpus):
    """
    Function to convert raw US election debate text
    and annotations into tagged character sequence

    Args:
        corpus: (list,list), which is the output of "read_us_election_corpus"

    Returns:
        list: each component contains annotated characters
    """
    # create empty list
    tagged = []
    # tag basic cases
    for i,raw in tqdm(enumerate(corpus[0]),total=len(corpus[0])):
        char_raw = list(raw)
        for j,char in enumerate(char_raw):
            if char not in ["\n","\r"]:
                char_raw[j] = "0"
        tagged.append(char_raw)
    # tag claims and premises
    for i in tqdm(range(len(tagged))):
        for annotation in corpus[1]:
            if annotation[0] == i:
                ann_type = annotation[2]
                spans = annotation[3]
                for z,pair in enumerate(spans):
                    for j in range(pair[0],pair[1]):
                        char = tagged[i][j]
                        if char not in ["\n","\r"]:
                            if ann_type == "Claim":
                                tagged[i][j] = "1"
                            elif ann_type == "Premise":
                                tagged[i][j] = "2"
    # join and return final tag
    return ["".join(char for char in segment) for segment in tqdm(tagged)]

def flatten(char_sequences):
    """
    Function to split and flatten character sequences

    Args:
        char_sequences: list containing character sequences

    Returns:
        list: flattened arguments character sequences
    """
    flat = []
    for speech in char_sequences:
        split = re.split(r"\r\n|\n|\r|\n\r",speech)
        for segment in split:
            if segment != "":
                flat.append(list(segment))
    return flat
