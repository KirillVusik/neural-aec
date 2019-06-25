import re

import numpy as np
from flask import Flask, abort, render_template, request, jsonify
from keras.models import load_model
from model import encode_expression, decode_result, ALPHABET,\
    OPERATIONS, MIN_NUMBER, MAX_NUMBER, MAX_EXPRESSION_LENGTH,\
    MAX_NUMBER_IN_EXPRESSION

app = Flask(__name__)
app.config.from_object('config')

model = load_model(app.config['MODEL_CHECKPOINT'])
model._make_predict_function()


@app.route('/')
def index():
    alphabet = ''.join(ALPHABET)
    escaped_alphabet = re.escape(''.join(ALPHABET))
    input_pattern = '^[{}]*$'.format(escaped_alphabet)
    return render_template(
        'index.html',
        pattern=input_pattern,
        alphabet=alphabet,
        operations=', '.join(OPERATIONS),
        min_number=MIN_NUMBER,
        max_number=MAX_NUMBER,
        max_numbers_in_expression=MAX_NUMBER_IN_EXPRESSION,
        max_length=MAX_EXPRESSION_LENGTH
    )


@app.route('/evaluate', methods=['POST'])
def evaluate():
    if not request.json or 'expression' not in request.json:
        abort(400)
    expression = request.json['expression']
    expression = ''.join(expression)
    validate_expression(expression)
    result = evalutate_using_model(expression)
    return jsonify({'result': '{}'.format(result)})


def validate_expression(expression):
    if set(expression) <= set(ALPHABET) and len(expression) < MAX_EXPRESSION_LENGTH:
        try:
            eval(expression)
            return  # OK
        except:
            pass
    abort(400)


def evalutate_using_model(expression):
    expression_vector = encode_expression(expression)
    # create a single-element batch
    expression_vector = np.expand_dims(expression_vector, axis=0)
    result_vector = model.predict(expression_vector)
    # get result from single-element batch
    return decode_result(result_vector[0])


if __name__ == '__main__':
    app.run()
