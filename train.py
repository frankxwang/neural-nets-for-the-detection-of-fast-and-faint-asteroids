from keras.applications import *
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import *
from keras import backend as K
from keras.utils import Sequence
from keras.regularizers import *
from keras.activations import *

import efficientnet.keras as efn

import tensorflow as tf
import glob
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from astropy.visualization import LinearStretch, ZScaleInterval, ImageNormalize

import skimage.transform

import sklearn.metrics

import sep
import scipy.stats as stats

import random

import wandb
from wandb.keras import WandbCallback

import numpy as np

import tables

import yaml

from collections import namedtuple

import pickle as pkl

size = 80

# set this to the directory of the Keras model if resuming a session
pretrained_file = None

# path to dataset from generate_batches
dataset = "/mnt/etdisk9/ztf_neos/training_examples/batches/filtered_short_streaks_no_overlapV2.h5"


# normalize for visualization
def normalize(arr):
    return ImageNormalize(arr, interval=ZScaleInterval(), stretch=LinearStretch())(arr)


class PyTablesSequences(Sequence):

    def __init__(self, x_arr, y_arr, batch_size, buffer_size=2048):

        assert buffer_size % batch_size == 0

        self.x, self.y = x_arr, y_arr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.next_load = 0

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        # print(idx)
        idx = idx * self.batch_size
        if idx >= self.next_load or idx < self.next_load - self.buffer_size:
            self.x_block = self.x[idx:idx + self.buffer_size]
            self.y_block = self.y[idx:idx + self.buffer_size]
            indexes = np.random.permutation(len(self.x_block))
            self.x_block = self.x_block[indexes]
            self.y_block = self.y_block[indexes]
            self.next_load = idx + self.buffer_size

        block_index = idx - (self.next_load - self.buffer_size)
        batch_x = self.x_block[block_index:block_index + self.batch_size]
        batch_y = self.y_block[block_index:block_index + self.batch_size]

        batch_x[~np.isfinite(batch_x)] = 0

        return batch_x, batch_y


def confusion_matrix(y_true, y_pred):
    y_pred_true = y_pred[y_true == 1]
    y_pred_false = y_pred[y_true == 0]

    correct = np.where(y_pred_true > 0.5)
    tp = len(correct[0])
    fn = len(y_pred_true) - tp

    correct = np.where(y_pred_false < 0.5)
    tn = len(correct[0])
    fp = len(y_pred_false) - tn

    return np.array([[tp, fp], [fn, tn]])


class WandbLogger(WandbCallback):
    def __init__(self, test_data=None, log_val_frequency=None, use_generators=False, train_gen=None, val_gen=None, **kwargs):
        super().__init__(**kwargs)

        self.using_test_data = test_data is not None
        if test_data is not None:
            self.test_data = test_data[0]
            self.test_labels = test_data[1]
        self.log_val_frequency = log_val_frequency
        self.best_fpr = 100000
        self.val_gen = val_gen

    def on_epoch_end(self, epoch, logs={}):
        super().on_epoch_end(epoch, logs)

        if self.using_test_data:
            self.test_labels = self.test_labels.flatten()
            predictions_real = self.model.predict(self.test_data).flatten()
            
            failed_indexes = (np.abs(predictions_real[:] - self.test_labels) > 0.5)

            test_data_failed = self.test_data[failed_indexes]
            test_labels_failed = self.test_labels[failed_indexes]
            preds_failed = predictions_real[failed_indexes]

            images = [wandb.Image(test_data_failed[i][..., 0],
                                  caption=str(preds_failed[i]) + " " + str(test_labels_failed[i]))
                      for i in range(len(test_labels_failed))]
            wandb.log({"real_failed_examples": images})

            images = [wandb.Image(self.test_data[::-1][i][..., 0], caption=str(predictions_real[::-1][i]) + " " + str(self.test_labels[::-1][i]))
                      for i in range(len(self.test_data))]
            wandb.log({"real_examples": images})

            wandb.log({"confusion matrix test": self.make_confusion_matrix((self.test_data, self.test_labels))})

    def make_confusion_matrix(self, data, return_arr=False):
        x, y_true = data[0], data[1]
        y_true = np.reshape(y_true, (len(y_true), 1))

        y_pred = self.model.predict(x)
        y_pred = np.reshape(y_pred, (len(y_pred), 1))

        c_matrix = confusion_matrix(y_true, y_pred).flatten()

        table = wandb.Table(["True Positive", "False Positive", "False Negative", "True Negative"], [c_matrix.tolist()])

        if return_arr:
            return table, c_matrix.tolist()

        return table

    # This is what keras used pre tensorflow.keras
    def on_train_batch_end(self, batch, logs=None):

        if not self._graph_rendered:
            wandb.run.summary['graph'] = wandb.Graph.from_keras(self.model)
            self._graph_rendered = True

        if self.log_batch_frequency and batch % self.log_batch_frequency == 0:
            wandb.log(logs, commit=True)

        if self.log_val_frequency and batch % self.log_val_frequency == 0 and batch != 0:

            val_labels = []
            val_predictions = []
            incorrect_classifications = []
            preds_failed = []
            labels_failed = []
            for i in range(len(self.val_gen)):
                batch_x, batch_y = self.val_gen[i]
                val_labels.append(batch_y)
                val_predictions.append(self.model.predict(batch_x, batch_size=len(batch_x)).flatten())
                if len(incorrect_classifications) < 500:
                    indicies = np.logical_and(val_predictions[-1] > 0.9, batch_y == 0)
                    incorrect_classifications.extend(batch_x[indicies])
                    preds_failed.extend(val_predictions[-1][indicies])
                    labels_failed.extend(batch_y[indicies])

            images = [wandb.Image(incorrect_classifications[i][..., 0],
                                  caption=str(preds_failed[i]) + " " + str(labels_failed[i]))
                      for i in range(len(incorrect_classifications))]
            wandb.log({"val_failed_examples": images})

            val_labels = np.concatenate(val_labels).flatten()
            val_predictions = np.concatenate(val_predictions).flatten()

            fpr, tpr, threshold = sklearn.metrics.roc_curve(val_labels, val_predictions)

            roc_auc = sklearn.metrics.roc_auc_score(val_labels, val_predictions)

            plt.figure()
            plt.plot(fpr, tpr, 'b', label='AUC = %0.10f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

            wandb.log({'roc_val_subset': plt})

            tp, fp, fn, tn = confusion_matrix(val_labels, val_predictions).flatten()

            table = wandb.Table(["True Positive", "False Positive", "False Negative", "True Negative"], [[tp, fp, fn, tn]])
            wandb.log({"confusion matrix val": table})
            wandb.log({"val_accuracy": (tp+tn)/(tp + tn + fp + fn)})

            tpr_index = min(np.searchsorted(tpr, 0.97), len(tpr) - 1)
            tpr_close = tpr[tpr_index]
            fpr_best = fpr[tpr_index]
            thresh = threshold[tpr_index]

            wandb.log({"false_positive_rate_0.5_cutoff": fp / (fp + tn),
                       "true_positive_rate_0.5_cutoff": tp / (tp + fn),
                       "tpr_97_tpr": tpr_close,
                       "fpr_97_tpr": fpr_best,
                       "thresh_97_tpr": thresh})

            if self.best_fpr > fpr_best:
                self.best_fpr = fpr_best
                filep = "/".join(self.filepath.split("/")[:-1])
                self.model.save(filep + "/model-best_" + str(fpr_best) + "_" + str(tpr_close) + "_" + str(thresh) + ".h5")

            wandb.log({"best_fpr_97_tpr": self.best_fpr})
            
fileh = tables.open_file(dataset, mode="r")

print("Loading Train Dataset")

train_images = fileh.root.train_images
train_labels = fileh.root.train_labels

print(len(indexes_filt), len(train_labels))
train_data = PyTablesSequences(train_images, train_labels, 32, buffer_size=1760)

print("Loading Val Dataset")

val_images = fileh.root.val_images
val_labels = fileh.root.val_labels

val_data = PyTablesSequences(val_images, val_labels, 32, buffer_size=640)

print("Training")

wandb.init(project="streak-detection")

if pretrained_file is None:
    images = Input((size, size, 2))

    x = efn.EfficientNetB1(input_shape=(size, size, 2), weights=None, include_top=False)(images)

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)

    output = Dense(1, activation="sigmoid")(x)

    model = Model(images, output)

    model.compile(optimizer=Adam(lr=0.0005), loss="binary_crossentropy", metrics=["accuracy"])
else:
    model = load_model(pretrained_file)
model.fit_generator(train_data, epochs=200, shuffle=False,
                    max_queue_size=50,
                    callbacks=[WandbLogger(monitor="val_loss",
                                           mode="min",
                                           log_batch_frequency=10,
                                           log_val_frequency=10000,
                                           val_gen=val_data,
                                           save_model=False,
                                           verbose=1)])
