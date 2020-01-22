#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import bert
import numpy as np
from tqdm import tqdm

def initialize_bert_tokenizer():
    model_name = "albert_base_v2"
    model_dir = bert.fetch_google_albert_model(model_name, ".models")
    spm = os.path.join(model_dir,"30k-clean.model")
    vocab = os.path.join(model_dir,"30k-clean.vocab")
    Tokenizer = bert.albert_tokenization.FullTokenizer(vocab,
                                                       spm_model_file=spm)
    return Tokenizer

def project_to_ids(train_data,label_id_map,max_seq_length=128):
    Tokenizer = initialize_bert_tokenizer()
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
