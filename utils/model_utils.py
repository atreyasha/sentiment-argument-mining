#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import bert
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Conv1D, LSTM, TimeDistributed, Input
from sklearn.metrics import classification_report


def class_acc(label_threshold_less):
    """
    Wrapper function to return keras accuracy logger

    Args:
        label_threshold_less (int): all label IDs strictly less than this number
        will be ignored in class accuracy calculations

    Returns:
        argument_candidate_acc (function)
    """
    def argument_candidate_acc(y_true, y_pred):
        """
        Function which returns argument candidate accuracy using
        the Keras backend

        Args:
            y_true (np.ndarray): true labels
            y_pred (np.ndarray): predicted labels

        Returns:
            class_accuracy (int): simple accuracy of argument candidates
        """
        class_id_true = K.cast(y_true, 'int64')
        class_id_preds = K.argmax(y_pred, axis=-1)
        accuracy_mask = K.cast(K.less(class_id_preds, label_threshold_less),
                               'float32')
        accuracy_mask = 1 - accuracy_mask
        class_acc_tensor = (
            K.cast(K.equal(class_id_true, class_id_preds), 'float32') *
            accuracy_mask)
        class_accuracy = (K.sum(class_acc_tensor) /
                          K.maximum(K.sum(accuracy_mask), 1))
        return class_accuracy

    return argument_candidate_acc


def class_report(y_true, y_pred, greater_than_equal=3):
    """
    Function to calculate a classification report for predictions

    Args:
        y_true (np.ndarray): true labels
        y_pred (np.ndarray): predicted labels
        greater_than_equal (int): all clases more than or equal to this
        number will be considered

    Returns:
        (dict): classification report for desired labels
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    relevant_indices = np.where(y_true >= greater_than_equal)[0]
    y_pred = y_pred[relevant_indices]
    y_true = y_true[relevant_indices]
    return classification_report(y_true, y_pred, output_dict=True)


def fetch_bert_layer():
    """
    Function to return ALBERT layer and weights

    Returns:
        l_bert (bert.model.BertModelLayer): BERT layer
        model_ckpt (str): path to best model checkpoint
    """
    model_name = "albert_base_v2"
    model_dir = bert.fetch_google_albert_model(model_name, ".models")
    model_ckpt = os.path.join(model_dir, "model.ckpt-best")
    model_params = bert.albert_params(model_dir)
    l_bert = bert.BertModelLayer.from_params(model_params, name="albert")
    return l_bert, model_ckpt


def learning_rate_scheduler(max_learn_rate, end_learn_rate, warmup_epoch_count,
                            total_epoch_count):
    """
    Wrapper function to return keras learning rate scheduler callback

    Args:
        max_learn_rate (float): maximum possible learning rate (achieved at peak
        of warmup epochs)
        end_learn_rate (float): minimum learning rate (achieved at end of
        maximum training epochs)
        warmup_epoch_count (int): number of epochs where learning rate rises
        total_epoch_count (int): maximum training epochs

    Returns:
        (tensorflow.keras.callbacks.LearningRateScheduler): LR-scheduler with
        internally passed learning rates
    """
    def lr_scheduler(epoch):
        """
        Output current learning rate based on epoch count, warmup and
        exponential decay

        Args:
            epoch (int): current epoch number

        Returns:
            (float): current learning rate
        """
        if epoch < warmup_epoch_count:
            res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate * math.exp(
                math.log(end_learn_rate / max_learn_rate) *
                (epoch - warmup_epoch_count + 1) /
                (total_epoch_count - warmup_epoch_count + 1))
        return float(res)

    return LearningRateScheduler(lr_scheduler, verbose=1)


def create_model(l_bert, model_ckpt, max_seq_len, num_labels,
                 label_threshold_less, model_type):
    """
    Wrapper function to return keras learning rate scheduler callback

    Args:
        l_bert (bert.model.BertModelLayer): BERT layer
        model_ckpt (str): path to best model checkpoint
        max_seq_length (int): maximum sequence length for training data
        num_labels (int): final output dimensionality per token
        label_threshold_less (int): all label IDs strictly less than this number
        will be ignored in class accuracy calculations
        model_type (str): type of model decoder to use, see
        './utils/model_utils.py'

    Returns:
        model (tensorflow.python.keras.engine.training.Model): final compiled
        model which can be used for fine-tuning
    """
    input_ids = Input(shape=(max_seq_len, ), dtype='int32')
    output = l_bert(input_ids)
    if model_type == "TD_Dense":
        output = TimeDistributed(Dense(512))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(256))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(128))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(64))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(num_labels))(output)
    elif model_type == "1D_CNN":
        output = Conv1D(512, 3, padding="same")(output)
        output = Activation("relu")(output)
        output = Conv1D(256, 3, padding="same")(output)
        output = Activation("relu")(output)
        output = Conv1D(128, 3, padding="same")(output)
        output = Activation("relu")(output)
        output = Conv1D(64, 3, padding="same")(output)
        output = Activation("relu")(output)
        output = Conv1D(num_labels, 3, padding="same")(output)
    elif model_type == "Stacked_LSTM":
        output = LSTM(512, return_sequences=True)(output)
        output = LSTM(256, return_sequences=True)(output)
        output = LSTM(128, return_sequences=True)(output)
        output = TimeDistributed(Dense(64))(output)
        output = Activation("relu")(output)
        output = TimeDistributed(Dense(num_labels))(output)
    prob = Activation("softmax")(output)
    model = tf.keras.Model(inputs=input_ids, outputs=prob)
    model.build(input_shape=(None, max_seq_len))
    bert.load_albert_weights(l_bert, model_ckpt)
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[class_acc(label_threshold_less)])
    model.summary()
    return model
