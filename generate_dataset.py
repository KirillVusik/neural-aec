import argparse
import math
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import OPERATIONS, MAX_EXPRESSION_LENGTH, MAX_RESULT_LENGTH,\
    MIN_NUMBER, MAX_NUMBER, MAX_NUMBER_IN_EXPRESSION, VECTOR_SIZE


class _Expression:
    OPS = OPERATIONS
    GROUP_PROB = 0.3
    MIN_NUM, MAX_NUM = MIN_NUMBER, MAX_NUMBER

    def __init__(self, max_numbers):
        self._expression = str(random.randint(MIN_NUMBER, MAX_NUMBER))
        numbers_count = random.randint(2, max_numbers)
        for _ in range(1, numbers_count):
            left = self._expression
            left = self._maybe_group(left)
            right = str(random.randint(MIN_NUMBER, MAX_NUMBER))
            right = self._maybe_group(right)
            # always group negatives on the right side
            if random.random() < 0.5:
                left, right = right, left
            if right.startswith('-'):
                right = '({})'.format(right)
            op = random.choice(self.OPS)
            self._expression = "{0}{1}{2}".format(left, op, right)

    def _maybe_group(self, expression):
        if (random.random() < self.GROUP_PROB):
            return '({})'.format(expression)
        else:
            return expression

    def __str__(self):
        return self._expression


def generate_expression():
    return str(_Expression(MAX_NUMBER_IN_EXPRESSION))


def train_test_generator(samples_count):
    for _ in(range(samples_count)):
        expression = generate_expression()
        while len(expression) > MAX_EXPRESSION_LENGTH:
            expression = generate_expression()

        result = str(eval(expression))
        yield expression, result


def get_args():
    parser = argparse.ArgumentParser(
        description='Generates dataset for training')

    parser.add_argument('-c', '--count',
                        type=int,
                        dest='samples_count',
                        required=True,
                        help='Count of (expression, result) pairs to generate')
    parser.add_argument('-o', '--output_path',
                        dest='output_path',
                        required=True,
                        help='Path to save the dataset')
    parser.add_argument('-s', '--seed',
                        type=int,
                        dest='seed',
                        help='Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    out_path = os.path.abspath(args.output_path)
    parent_dir = os.path.dirname(out_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if args.seed is not None:
        random.seed(args.seed)
    X, Y = zip(*tqdm(train_test_generator(args.samples_count),
                     total=args.samples_count))
    data = {'X': X, 'Y': Y}
    dataframe = pd.DataFrame(data)

    print('Saving to {}'.format(out_path))
    dataframe.to_hdf(out_path, key='train_data',
                     mode='w', format='fixed')
