#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

############################
# define key functions
############################

def read_raw_corpus():
    # read raw text into memory by sorting first through indices
    files_raw = glob("./data/USElectionDebates/*.txt")
    indices = [int(re.sub(r".*\/(\d+)_(\d+).*","\g<1>",File)) for File in files_raw]
    files_raw = [f for i,f in sorted(zip(indices,files_raw))]
    files_ann = glob("./data/USElectionDebates/*.ann")
    indices = [int(re.sub(r".*\/(\d+)_(\d+).*","\g<1>",File)) for File in files_ann]
    files_ann = [f for i,f in sorted(zip(indices,files_ann))]
    # read raw text into pythonic list
    raw = []
    for File in tqdm(files_raw):
        with open(File,"r",encoding="utf-8") as f:
            text = f.read().rstrip()
            text = re.sub(" +"," ",text)
            raw.append(re.sub("\s*\n\s*","\n",text))
    # read annotations into pandas dataframe
    anom = []
    ann_df = pd.DataFrame(columns=["index", "id", "type", "min_bound",
                                "max_bound", "text"])
    dubious_count = 0
    for i,File in tqdm(enumerate(files_ann),total=len(files_ann)):
        with open(File,"r",encoding="utf-8") as f:
            ann_data = f.readlines()
        for annotation in ann_data:
            hold = annotation.replace("\n","").split("\t")
            hold_type = hold[1].split(" ")[0]
            findings = list(re.finditer(re.escape(hold[2]),raw[i]))
            if len(findings) == 1:
                min_bound = findings[0].span()[0]
                max_bound = findings[0].span()[1]
            else:
                min_bound = None
                max_bound = None
                anom.append((hold,i))
                dubious_count += 1
            ann_df = ann_df.append({"index":i,"id":hold[0],"type":hold_type,
                           "min_bound":min_bound, "max_bound":max_bound,
                                    "text":hold[2]},ignore_index=True)
    print("Problematic entries: %s" % str(dubious_count))
    return raw,ann_df,anom

# TODO deal with skip connection components by simply extending them, would still prove problematic
# TODO check for exact nature of all anomalies, if argument is problematic due to more than one occurance, choose that which is nearer
# TODO if sentence ends with a period and is not included, include it in component
# TODO or perhaps switch completely over to persuasive essay corpus which might be similar
# check proportion of anomalous issues
# look up nested argument structures, look up persuasive essay corpus
# maybe get rid of contextual nouns altogether to keep model unbiased
# add different classes in unknown vocabulary -> such as unknown noun, unknown adjective etc.

# for presentation, talk about single task and joint task goals in terms of preference
# ideallly use us election data, but it is not very user-friendly and perhaps not formal enough
# try to use persuasive essay corpus as it is more objective in terms of language
# attempt this task and we can check our results, potentially to check if we can do a full joint task

# TODO develop basic or joint pipeline on persuasive essay corpus and then try expanding
# TODO recreate corpus (remove double spacing and make all indices consisten)
# TODO split cleanly and output type and indices
# TODO check/assert that annotation text equals corpus text
# TODO put into efficient online data structure such as pd dataframe
# TODO make dual datasets with simple difference and joint double view
# TODO make character and word representations with both cased and non-cased views

def extract_text(speech,min_bound,max_bound):
    pass

# TODO claim/premise corresponds to same list index minus 1 on each
# TODO need to spit into smaller speech segments or paragraphs to pass into pipeline
# might conflict with list indexing; find efficient workaround within pipeline for this

############################
# comments/to-do's
############################

# find effective idea of how to split into claims/premises with tree structure
# from then proceed to encoding it into vocabulary, might need BERT indices for this
# ignore us-politics specific terms to not convolute the vocabulary, but can be addiitonal step
# might be useful to perform NER on this dataset to isolate these objects
