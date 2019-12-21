#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pre_process import *
from keras_transformer import get_model, decode

def get_corpus_input(subtype="char"):
    corpus = read_us_election_corpus()
    tagged = char_tag(corpus)
    flat_text = flatten(corpus[0])
    flat_ann = flatten(tagged)
    return flat_text,flat_ann

def build_token_dict(token_list):
    token_dict = {
        '<PAD>': 0,
        '<START>': 1,
        '<END>': 2,
    }
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict

def get_transformer_params():
    source_tokens,target_tokens = get_corpus_input()
    source_token_dict = build_token_dict(source_tokens)
    target_token_dict = build_token_dict(target_tokens)
    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}
    # add special tokens
    encode_tokens = [['<START>'] + tokens + ['<END>']
                     for tokens in source_tokens]
    decode_tokens = [['<START>'] + tokens + ['<END>']
                     for tokens in target_tokens]
    output_tokens = [tokens + ['<END>', '<PAD>']
                     for tokens in target_tokens]
    # padding
    source_max_len = max(map(len, encode_tokens))
    target_max_len = max(map(len, decode_tokens))
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
    encode_input = [list(map(lambda x: source_token_dict[x], tokens))
                    for tokens in tqdm(encode_tokens)]
    decode_input = [list(map(lambda x: target_token_dict[x], tokens))
                    for tokens in tqdm(decode_tokens)]
    decode_output = [list(map(lambda x: [target_token_dict[x]], tokens))
                    for tokens in tqdm(output_tokens)]
    return (encode_input,decode_input,decode_output,
            source_token_dict,target_token_dict)

(encode_input,decode_input,decode_output,
 source_token_dict,target_token_dict) = get_transformer_params()
# TODO save numpy binary for later use
# TODO OOM issues when using character models, perhaps convert to smaller
# TODO or try simple tokenization approach instead of complicated one

# Build & fit model
model = get_model(
    token_num=max(len(source_token_dict), len(target_token_dict)),
    embed_dim=32,
    encoder_num=2,
    decoder_num=2,
    head_num=4,
    hidden_dim=128,
    dropout_rate=0.05,
    use_same_embed=False,  # Use different embeddings for different languages
)
model.compile('adam', 'sparse_categorical_crossentropy')
model.summary()

model.fit(
    x=[np.array(encode_input), np.array(decode_input)],
    y=np.array(decode_output),
    epochs=10,
    batch_size=32,
)
