import argparse
import os
from collections import namedtuple
from itertools import product

import numpy as np
import pandas as pd
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint,\
    ReduceLROnPlateau
from tqdm import tqdm

from model import MAX_EXPRESSION_LENGTH, MAX_RESULT_LENGTH, VECTOR_SIZE,\
    build_model, encode_expression, encode_result


# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# Uncomment for monitoring the GPU usage
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# set_session(sess)


RNN_TYPES = ['LSTM', 'GRU']
INPUT_LAYERS_COUNT = [1]
OUTPUT_LAYERS_COUNT = [1]
INPUT_UNITS_COUNT = [100]
OUTPUT_UNITS_COUNT = [100]
EPOCHS = 50
BATCH_SIZE = 512
VALIDATION_SPLIT = 0.1


def prepare_data(dataset_path):
    dataset = pd.read_hdf(dataset_path)
    X = dataset.X.values
    X = np.array([encode_expression(x) for x in X])
    Y = dataset.Y.values
    Y = np.array([encode_result(y) for y in Y])
    print('Data has been prepared')
    return X, Y


def get_args():
    parser = argparse.ArgumentParser(
        description='Trains models')

    parser.add_argument('-d', '--dataset_path',
                        dest='dataset_path',
                        required=True,
                        help='Path to the train dataset')
    parser.add_argument('-o', '--output_dir',
                        dest='output_dir',
                        required=True,
                        help='Path to a folder to save trained models in')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        raise ValueError('Dataset does not exist')
    save_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    X, Y = prepare_data(dataset_path)
    for parameters in product(RNN_TYPES, INPUT_LAYERS_COUNT, INPUT_UNITS_COUNT,
                              OUTPUT_LAYERS_COUNT, OUTPUT_UNITS_COUNT):
        model = build_model(*parameters)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(0.001), metrics=['accuracy'])
        print(model.summary())

        early_stopping = EarlyStopping(
            monitor='val_loss', patience=3, verbose=1, mode='min')
        checkpoint_path = os.path.join(
            save_dir,
            model.name + '-{epoch:03d}-{acc:03f}-{val_acc:03f}.checkpoint.hdf5'
        )
        checkpoint = ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor='val_loss',
            mode='min'
        )
        reduce_loss = ReduceLROnPlateau(patience=2)
        model.fit(X, Y, batch_size=BATCH_SIZE, shuffle=True,
                  validation_split=VALIDATION_SPLIT, epochs=EPOCHS,
                  callbacks=[early_stopping, checkpoint, reduce_loss])
