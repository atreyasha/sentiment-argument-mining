#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import datetime
import bert
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from obj.arg_metav_formatter import *
import logging

# read in data as string representation
# map labels to integers, pad labels as well
# TODO use kamalkraj's implementation for data formatting, use local keras library for model handling
# TODO need to modify end of system to handle masking
# TODO replace local tokenizers with those suggested in bert-tf-2
# TODO issue of trimming/padding before or inside model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

def readfile(filename):
    """
    read file
    """
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])
    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["C", "P", "N", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

# TODO modify tokenizer here
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list, 1)}
    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(True)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, True)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(True)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [True] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(False)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(False)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features

model_name = "albert_base_v2"
model_dir    = bert.fetch_google_albert_model(model_name, ".models")
model_ckpt   = os.path.join(model_dir, "model.ckpt-best")
model_params = bert.albert_params(model_dir)
l_bert = bert.BertModelLayer.from_params(model_params, name="albert")

import sentencepiece as spm
spm_model = "./.models/albert_base_v2/albert_base/30k-clean.model"
sp = spm.SentencePieceProcessor()
sp.load(spm_model)
do_lower_case = True

processor = NerProcessor()
label_list = processor.get_labels()
num_labels = len(label_list) + 1
tokenizer = bert.albert_tokenization.FullTokenizer("./.models/albert_base_v2/albert_base/30k-clean.vocab", do_lower_case=True,spm_model_file="./.models/albert_base_v2/albert_base/30k-clean.model")

train_examples = processor.get_train_examples("./data/pre-processed/task_1/tokens")
label_map = {i: label for i, label in enumerate(label_list, 1)}
train_features = convert_examples_to_features(
    train_examples, label_list, 300, tokenizer)

train_x = np.array([input_set.input_ids for input_set in train_features])
train_y = np.array([input_set.label_id for input_set in train_features])

def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/
                                                max_learn_rate)*
                                        (epoch-warmup_epoch_count+1)/
                                        (total_epoch_count-
                                        warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,
                                                                       verbose=1)
    return learning_rate_scheduler

def create_model(l_bert,model_ckpt,max_seq_len=300):
    """Creates a classification model."""
    input_ids      = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    output         = l_bert(input_ids)
    print("bert shape", output.shape)
    cls_out = tf.keras.layers.Dropout(0.5)(output)
    logits = tf.keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = tf.keras.layers.Dropout(0.5)(logits)
    logits = tf.keras.layers.Dense(units=1, activation="softmax")(logits)
    model = tf.keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))
    bert.load_albert_weights(l_bert, model_ckpt)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")])
    model.summary()
    return model

model = create_model(l_bert,model_ckpt)
total_epoch_count=3

model.fit(x=train_x,y=train_y,
          validation_split=0.1,
          batch_size=48,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=20,
                                                    total_epoch_count=total_epoch_count),
                     tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)])
model.save_weights('./test.h5', overwrite=True)
