import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, CuDNNLSTM, CuDNNGRU, LSTM, GRU,\
    RepeatVector, BatchNormalization
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.utils import to_categorical

OPERATIONS = list('+-')
ALPHABET = list(map(str, range(10))) + OPERATIONS + list('() ')
VECTOR_SIZE = len(ALPHABET)
MAX_EXPRESSION_LENGTH = 16
MAX_NUMBER = 99
MIN_NUMBER = -99
MAX_NUMBER_IN_EXPRESSION = 4
MAX_RESULT_LENGTH = 4


class _OneHotEncoder():
    def __init__(self, alphabet):
        self._alphabet = alphabet
        self._encoded_chars = to_categorical(range(len(alphabet)))
        self._char_to_index = {
            char: index for index, char in enumerate(alphabet)}

    def encode(self, expression):
        # Matrix of vector rows
        return np.array(list(map(self._encode_char, expression)), dtype=bool)

    def decode(self, matrix):
        return ''.join(np.apply_along_axis(self._decode_char, 1, matrix))

    def _encode_char(self, char):
        index = self._char_to_index[char]
        return self._encoded_chars[index]

    def _decode_char(self, vector):
        index = np.argmax(vector)
        return self._alphabet[index]


encoder = _OneHotEncoder(ALPHABET)


def build_model(rnn_type, encode_layers_count, encode_units_count,
                decode_layers_count, decode_units_count):
    if rnn_type == 'LSTM':
        RNN = LSTM
    elif rnn_type == 'GRU':
        RNN = GRU
    else:
        raise ValueError(
            'RNN_type expected to be LSTM or GRU, got: {}'.format(rnn_type))
    input_layer = Input(shape=(MAX_EXPRESSION_LENGTH, VECTOR_SIZE))
    # encoder
    output_layer = input_layer
    for _ in range(encode_layers_count - 1):
        output_layer = Bidirectional(RNN(encode_units_count,
                                     return_sequences=True))(output_layer)
    output_layer = Bidirectional(RNN(encode_units_count))(output_layer)

    output_layer = RepeatVector(MAX_RESULT_LENGTH)(output_layer)
    # decoder
    for _ in range(decode_layers_count):
        output_layer = Bidirectional(RNN(decode_units_count,
                                         return_sequences=True))(output_layer)

    output_layer = TimeDistributed(
        Dense(VECTOR_SIZE, activation='softmax'))(output_layer)
    rnn_stack_description = '{}-layers-{}-units-'
    encoder_description = rnn_stack_description.format(
        encode_layers_count, encode_units_count) + 'encoder'
    decoder_description = rnn_stack_description.format(
        decode_layers_count, decode_units_count) + 'decoder'
    model = Model(inputs=input_layer, outputs=output_layer)
    model.name = '{}_{}_{}'.format(
        rnn_type, encoder_description, decoder_description)
    return model


def encode_sequence(sequence, required_length):
    # remove whitespaces
    sequence = "".join(sequence.split())
    sequence = sequence.rjust(required_length)
    return encoder.encode(sequence)


def encode_expression(expressions):
    return encode_sequence(expressions, MAX_EXPRESSION_LENGTH)


def encode_result(result):
    return encode_sequence(result, MAX_RESULT_LENGTH)


def decode_result(result_vector):
    result = encoder.decode(result_vector)
    return result.strip()
