#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from pre_process import *

# TODO add calls to R scripts for plotting
# TODO consider combining this aspect into pre_process workflow

def corpus2tokenplot(directory="./data/USElectionDebates/corpus/"):
    corpus = read_us_election_corpus()
    tagged = char_tag(corpus,spaces=True)
    indices,flat_text = flatten(corpus[0],indices=True)
    flat_ann = flatten(tagged)
    assert len(flat_text) == len(flat_ann)
    flat_text,flat_ann = correct_periods(flat_text,flat_ann,spaces=True)
    collection = []
    print("splitting and tokenizing sentences")
    Tokenizer = initialize_bert_tokenizer()
    # ensure nltk sentence tokenizer is present
    try:
        nltk.tokenize.sent_tokenize("testing. hello")
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
    # convert collection into debate wise tokenized set
    new_collection = []
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
