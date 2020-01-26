#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score

def class_acc(label_threshold_less):
    def argument_candidate_acc(y_true, y_pred):
        class_id_true = K.cast(y_true,'int64')
        class_id_preds = K.argmax(y_pred, axis=-1)
        accuracy_mask = K.cast(K.less(class_id_preds,label_threshold_less),
                               'float32')
        accuracy_mask = 1 - accuracy_mask
        class_acc_tensor = (K.cast(K.equal(class_id_true,
                                          class_id_preds),
                                   'float32') * accuracy_mask)
        class_accuracy = (K.sum(class_acc_tensor)/
                          K.maximum(K.sum(accuracy_mask),1))
        return class_accuracy
    return argument_candidate_acc

class FilteredF1Score(tf.keras.callbacks.Callback):
    def __init__(self, to_ignore=[0,1,2]):
        self.to_ignore = to_ignore

    def on_train_begin(self, logs={}):
        self.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        y_true = self.validation_data[1]
        y_pred = np.argmax(self.model.predict(self.validation_data[0]), axis=-1)
        val_f1 = filtered_f1(y_true, y_pred)
        self.val_f1s.append(val_f1)
        print(" - val_f1: %f" % (val_f1))

    def filtered_f1(self, y_true, y_pred):
        y_pred_flat, y_true_flat = [], []
        for y_pred_i, y_true_i in zip(y_pred.flatten(), y_true.flatten()):
            if y_true_i not in self.to_ignore:
                y_pred_flat.append(y_pred_i)
                y_true_flat.append(y_true_i)
        result = f1_score(y_true_flat, y_pred_flat, average='macro')
        return result
