#!/usr/bin/env python

import argparse
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras import optimizers
from keras.layers import Conv3D, Dense, Dropout, Flatten, MaxPooling3D, ZeroPadding3D
from keras.models import Sequential
import numpy as np
import sklearn
import tensorflow as tf

from callbacks import TrackTrainHistory, TrackTestHistoryAndModel
import data

import warnings
warnings.filterwarnings('ignore', category=sklearn.exceptions.UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


LEARNING_RATE = 1.0
MOMENTUM = 0.0
DECAY = 5e-6
L2_FACTOR = 0.0
DROPOUT = 0.0
ACTIVATION = 'relu'
LOSS = 'mse'
ACCURACY = 'accuracy'


def build_standard_model(input_shape, class_count, activation, dropout_fraction):
    """
    Creates a 3D convolutional neural network with three filter layers and two prediction layers.

    :param input_shape:         tuple(int), four-dimensional tuple of input batch size, usually (batch, window, window, 1)
    :param class_count:         int, the number of classes
    :param activation:          str, the used activation function for the network
    :param dropout_fraction:    float, the fraction level of dropout applied dropout regularization
    :return: model:             keras.Sequential, the constructed model
    """
    model = Sequential()

    model.add(Conv3D(48, kernel_size=(3, 3, 5), activation=activation, input_shape=input_shape))
    model.add(Dropout(dropout_fraction))
    model.add(MaxPooling3D(pool_size=(1, 1, 3)))
    model.add(ZeroPadding3D((0, 0, 2), data_format=None))

    model.add(Conv3D(32, kernel_size=(3, 3, 5), activation=activation))
    model.add(Dropout(dropout_fraction))
    model.add(MaxPooling3D(pool_size=(1, 1, 3)))
    model.add(ZeroPadding3D((0, 0, 2), data_format=None))

    model.add(Conv3D(32, kernel_size=(3, 3, 5), activation=activation))
    model.add(Dropout(dropout_fraction))
    model.add(MaxPooling3D(pool_size=(1, 1, 2)))
    model.add(Flatten())

    model.add(Dense(128, activation=activation))
    model.add(Dropout(dropout_fraction))
    model.add(Dense(128, activation=activation))
    model.add(Dense(class_count, activation='softmax'))

    return model


def make_patches(data, window):
    """
    Creates a five-dimensional tensor (height, width, window, window, channels) from the three-dimensional input data
    (height, width, channels). Has minimal memory footprint as the patches are created using striding indirection.

    :param data:        np.array(height, width, channels), three-dimensional raw input data
    :param window:      int, the square window size
    :return: patches:   np.array(height, width, window, window, channels), the five-dimensional output tensor
    """
    half_window = window // 2
    channels = data.shape[-1]
    padded_data = np.pad(data, ((half_window, half_window,), (half_window, half_window,), (0, 0,)), mode='edge')

    # generate the windows using a strided view
    out_shape = (data.shape[0], data.shape[1], 1, window, window, channels,)
    out_strides = 2 * padded_data.strides

    return np.lib.stride_tricks.as_strided(padded_data, shape=out_shape, strides=out_strides)


def patch_generator(patches, labels, selection, items, batch_size, shuffle):
    """
    Generator for the on the fly extraction of patch batches during training, evaluation or prediction.

    :param patches:     np.array(height, width, window, window, channels), the five-dimensional patch tensor
    :param window:      np.array(height, width), two-dimensional label map
    :param selection:   np.array(height, width), two-dimensional boolean map of indices to generate from
    :param items:       int, number of items in the generator, works batch-adjusted, i.e. skips elements that do not
                        fill up towards a complete batch
    :param batch_size:  int, the size of the patch batch to be generated
    :param shuffle:     bool, flag indicating whether the generator should shuffle the input data after exhaustion
    :return: batch:     np.array(batch_size, window, window, channels), np.array(batch_size) the generated patch batch
                        and the corresponding labels
    """
    y, x = np.where(selection)
    patch_shape = (batch_size, *patches.shape[-3:], 1,)
    label_shape = (batch_size, labels.shape[-1],)

    while True:
        for i in range(batch_size, items, batch_size):
            batch_y, batch_x = y[i - batch_size:i], x[i - batch_size:i]
            yield patches[batch_y, batch_x].reshape(*patch_shape), labels[batch_y, batch_x].reshape(*label_shape)

        if shuffle:
            indices = np.arange(y.shape[0])
            np.random.shuffle(indices)
            y, x = y[indices], x[indices]


def train_network(arguments):
    """
    Trains the entire network. Entails: data loading, preprocessing, model construction, patch generation, batched
    training with the data, model prediction and metric calculation.

    :param arguments:   dict, the configuration arguments for the training read from the command line
    """
    # set the random seed for reproducibility
    np.random.seed(arguments.seed)
    tf.set_random_seed(arguments.seed)

    # begin network training
    start_time = time.time()
    print('__Training: {}__'.format(arguments.in_file), flush=True)

    # load a patches dataset
    print('\tLoading data... ', end='', flush=True)
    raw_data, labels = data.read_datasets(arguments.in_file, data.DATA, data.LABELS)
    selection, mean, stddev = data.read_datasets(arguments.mask, data.SELECTION, data.MEAN, data.STDDEV)
    print('\t[Done, took: {:.2f}s]'.format(time.time() - start_time))

    # data preprocessing
    preprocessing_time = time.time()
    print('\tPreprocessing data... ', end='', flush=True)
    class_count = np.unique(labels).shape[0]
    raw_data = (raw_data - mean) / stddev
    label_matrix = keras.utils.to_categorical(labels, num_classes=class_count)
    print('\t[Done, took: {:.2f}s]'.format(time.time() - preprocessing_time))

    # create the window patches
    patch_time = time.time()
    print('\tGenerating patches... ', end='', flush=True)
    window = arguments.window
    channels = raw_data.shape[-1]
    patches = make_patches(raw_data, window)
    print('\t[Done, took: {:.2f}s]'.format(time.time() - patch_time))

    # construct the model
    model_time = time.time()
    print('\tBuilding the network... ', end='', flush=True)
    with tf.device('/gpu:0' if arguments.gpu else '/cpu:0'):
        model = build_standard_model((window, window, channels, 1,), class_count, ACTIVATION, DROPOUT)
        # resume from previous weight checkpoint
        if arguments.checkpoint:
            model.load_weights(arguments.checkpoint)

        # compile the model with a standard gradient descent optimizer
        model.compile(
            optimizer=optimizers.SGD(lr=LEARNING_RATE, momentum=MOMENTUM, decay=DECAY),
            loss=LOSS, metrics=[ACCURACY]
        )

    # add tracking callbacks
    callbacks = [TrackTrainHistory(arguments.train_history)]
    if arguments.test:
        test_selection = selection == data.TEST
        test_items = test_selection.astype(np.uint8).sum()

        callbacks.append(TrackTestHistoryAndModel(
            patch_generator(patches, label_matrix, test_selection, test_items, arguments.batch, shuffle=False),
            test_items // arguments.batch, arguments.model, arguments.test_history)
        )
    print('[Done, took: {:.2f}s]'.format(time.time() - model_time))

    # actually start training the model
    training_start_time = time.time()
    print('\tTraining the model...\n', flush=True)
    train_selection = selection == data.TRAIN
    train_items = train_selection.astype(np.uint8).sum()
    model.fit_generator(
        patch_generator(patches, label_matrix, train_selection, train_items, arguments.batch, shuffle=True),
        steps_per_epoch=train_items // arguments.batch,
        epochs=arguments.epochs,
        callbacks=callbacks,
        shuffle=False
    )
    print('\n\t[Done, took: {:.2f}s]'.format(time.time() - training_start_time))

    # prediction on the test set
    prediction_start_time = time.time()
    print('\tPrediction... ', end='', flush=True)
    pred_selection = selection == data.TEST
    pred_items = pred_selection.astype(np.uint8).sum()
    test_prediction = model.predict_generator(
        patch_generator(patches, label_matrix, pred_selection, pred_items, batch_size=1, shuffle=False),
        steps=pred_items
    )
    print('\t\t[Done, took: {:.2f}s]'.format(time.time() - prediction_start_time))

    # compute the metrics
    metrics_start_time = time.time()
    print('\tMetrics...', end='', flush=True)
    most_probable_prediction = np.argmax(test_prediction, axis=1)
    test_labels = labels[pred_selection]

    confusion_matrix = sklearn.metrics.confusion_matrix(most_probable_prediction, test_labels).astype(np.float32)
    oa = sklearn.metrics.accuracy_score(most_probable_prediction, test_labels)
    kappa = sklearn.metrics.cohen_kappa_score(most_probable_prediction, test_labels)
    f1 = sklearn.metrics.f1_score(most_probable_prediction, test_labels, average=None)
    pa = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0).T
    aa = np.mean(pa)

    # memorize the metrics history
    np.set_printoptions(precision=2, threshold=np.iinfo(np.int64).max)
    with open(arguments.results, 'w') as file:
        file.writelines([
            'OA: {}\n'.format(oa),
            'AA: {}\n'.format(aa),
            'kappa: {}\n'.format(kappa),
            'F1: {}\n'.format(np.mean(f1)),
            '',
            'Accuracy for each class:\n',
            str(pa),
            '\nF1 score for each class:\n',
            str(f1),
            '\nConfusion matrix:\n',
            str(confusion_matrix)
        ])
    print('\t\t[Done, took: {:.2f}s]'.format(time.time() - metrics_start_time))


def positive_int(value):
    try:
        parsed = int(value)
        if not parsed > 0:
            raise ValueError()
        return parsed
    except ValueError:
        raise argparse.ArgumentTypeError('value must be an positive integer')


def positive_odd_int(value):
    value = positive_int(value)
    if value % 2 == 1:
        return value
    raise argparse.ArgumentTypeError('value must be an odd')


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        metavar='FILE',
        type=str,
        dest='in_file',
        help='HDF5 file containing the raw data'
    )
    parser.add_argument(
        metavar='MASK',
        type=str,
        dest='mask',
        help='HDF5 file containing the selection mask'
    )
    parser.add_argument(
        '-c', '--checkpoint',
        nargs='?',
        type=str,
        action='store',
        help='path to previous model HDF5 storage location'
    )
    parser.add_argument(
        '-m', '--model',
        nargs='?',
        type=str,
        action='store',
        const='model.h5',
        default='model.h5',
        help='path to model HDF5 storage location'
    )
    parser.add_argument(
        '-t', '--test',
        action='store_true',
        help='flag whether to validate model on test data'
    )
    parser.add_argument(
        '--train-history',
        nargs='?',
        type=str,
        action='store',
        const='train.csv',
        default='train.csv',
        help='path to training history CSV file storage location'
    )
    parser.add_argument(
        '--test-history',
        nargs='?',
        type=str,
        action='store',
        const='test.csv',
        default='test.csv',
        help='path to test history CSV file storage location'
    )
    parser.add_argument(
        '-r', '--results',
        nargs='?',
        type=str,
        action='store',
        const='results.csv',
        default='results.csv',
        help='path to result metric CSV file storage location'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=positive_int,
        action='store',
        default=500,
        help='training epochs'
    )
    parser.add_argument(
        '-b', '--batch',
        type=positive_int,
        action='store',
        default=50,
        help='window size for the square patches'
    )
    parser.add_argument(
        '-w', '--window',
        type=positive_odd_int,
        action='store',
        default=7,
        help='window size for the square patches'
    )
    parser.add_argument(
        '-g', '--gpu',
        action='store_true',
        help='flag indicating GPU utilization for training'
    )
    parser.add_argument(
        '-s', '--seed',
        type=positive_int,
        action='store',
        default=0,
        help='seed for the initialization of the network'
    )

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    train_network(arguments)
