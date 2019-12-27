#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import numpy as np
from keras_transformer import get_model, decode

def retrieve_data(dtype="tokens", file_path = "./data/pre-processed/"):
    """
    Function to retrieve data components to train transformer

    Args:
        dtype (str): whether to retrieve 'tokens' or 'char' data
        file_path (str): base file directory on which to find input

    Returns:
        encode_input (numpy.ndarray): encoded input tokens
        decode_input (numpy.ndarray): decoded input tokens
        decode_output (numpy.ndarray): decoded output tokens
        source_token_dict (dict): source tokens and indices
        target_token_dict (dict): target tokens and indices
    """
    if dtype == "tokens":
        prefix = "tokenized"
    elif dtype == "char":
        prefix == "characterized"
    with open(file_path+prefix+"_source_dict.json","r") as f:
        source_token_dict = json.load(f)
    with open(file_path+prefix+"_target_dict.json","r") as f:
        target_token_dict = json.load(f)
    encode_input = np.load(file_path+prefix+"_encode_input.npy")
    decode_input = np.load(file_path+prefix+"_decode_input.npy")
    decode_output = np.load(file_path+prefix+"_decode_output.npy")
    return (encode_input,decode_input,decode_output,
            source_token_dict,target_token_dict)

def train(dtype="tokens",
          epochs=50,batch_size=5,file_path="./models/"):
    """
    Function to run simple transformer and save final model

    Args:
        dtype (str): whether to retrieve 'tokens' or 'char' data
        epochs (int): maximum number of epochs used for training
        batch_size (int): batch size used in stochastic gradient descent;
                          it is recommended to keep this low (~5-10 batches)
                          to prevent GPU out-of-memory issues
        file_path (str): base file directory where h5-models will be stored
    """
    # get input data
    (encode_input,decode_input,decode_output,
            source_token_dict,target_token_dict) = retrieve_data(dtype)
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
    model.compile('adam','sparse_categorical_crossentropy')
    model.fit(x=[encode_input,decode_input],
              y=decode_output,epochs=epochs,batch_size=batch_size)
    model.save(file_path+dtype+"_single_run.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class
                                     =argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dtype", type=str, default="tokens",
                        help="whether to train model on 'tokens' or 'char'")
    parser.add_argument("--epochs", type=int, default=50,
                        help="maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="batch size in stochastic gradient descent")
    args = parser.parse_args()
    train(args.dtype,args.epochs,args.batch_size)
