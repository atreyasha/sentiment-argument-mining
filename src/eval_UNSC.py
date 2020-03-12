#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from pre_process_UNSC import *
from utils.model_utils import *

def load_model(direct):
    l_bert, model_ckpt = fetch_bert_layer()
    model_path = glob(os.path.join(direct,"*h5"))[0]
    model = load_model(model_path,
                       custom_objects={"BertModelLayer":l_bert,
                                       "argument_candidate_acc":class_acc(3)})
    y_pred = model.predict(test_X)
    y_pred = np.argmax(y_pred,axis=-1)
    out_dict = report(test_Y,y_pred)
