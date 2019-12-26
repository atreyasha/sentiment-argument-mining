#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from keras_transformer import get_model, decode

def retrieve_data():
    file_path = "./data/pre-processed/"
    with open(file_path+"tokenized_source_dict.json","r") as f:
        source_token_dict = json.load(f)
    with open(file_path+"tokenized_target_dict.json","r") as f:
        target_token_dict = json.load(f)
    encode_input = np.load(file_path+"tokenized_encode_input.npy")
    decode_input = np.load(file_path+"tokenized_decode_input.npy")
    decode_output = np.load(file_path+"tokenized_decode_output.npy")
    return (encode_input,decode_input,decode_output,
            source_token_dict,target_token_dict)

def train():
    # get input data
    (encode_input,decode_input,decode_output,
            source_token_dict,target_token_dict) = retrieve_data()
    # Build & fit model
    model = get_model(
        token_num=max(len(source_token_dict), len(target_token_dict)),
        embed_dim=32,
        encoder_num=2,
        decoder_num=2,
        head_num=4,
        hidden_dim=128,
        dropout_rate=0.05,
        use_same_embed=False)
    model.compile('adam', 'sparse_categorical_crossentropy')
    model.fit(x=[encode_input, decode_input],
        y=decode_output,
        epochs=10,
        batch_size=32)
    model.save("test.h5")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class
    #                                  =argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--dtype", type=str, default="tokens",
    #                     help="which type of data pre-processing;"+
    #                     " either 'tokens', 'char' or 'both'")
    # args = parser.parse_args()
    train()
