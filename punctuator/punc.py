#!/usr/bin/env python
# coding: utf-8
from __future__ import division, print_function

import os
import sys
import re
import logging
import pickle
from io import open, StringIO

import theano
import theano.tensor as T
import numpy as np

import gdown

from . import models
from . import data
from .convert_to_readable import convert

PUNCTUATOR_DATA_DIR = os.path.expanduser(os.environ.get('PUNCTUATOR_DATA_DIR', '~/.punctuator'))

MAX_SUBSEQUENCE_LEN = 200

# pylint: disable=redefined-outer-name

DEMO_DATA_GID = '0B7BsN5f2F1fZd1Q0aXlrUDhDbnM' # Demo-Europarl-EN.pcl


def download_model(gid=DEMO_DATA_GID):
    _cwd = os.getcwd()
    try:
        os.makedirs(PUNCTUATOR_DATA_DIR, exist_ok=True)
        os.chdir(PUNCTUATOR_DATA_DIR)
        logging.info('Downloading %s...', gid)
        fn = gdown.download(url=f'https://drive.google.com/uc?id={gid}', output=None, quiet=False)
        return os.path.join(PUNCTUATOR_DATA_DIR, fn)
    finally:
        os.chdir(_cwd)


def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T


def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    return punct_token[0]


def restore_with_pauses(output_file, text, pauses, word_vocabulary, reverse_punctuation_vocabulary, predict_function):
    i = 0
    if isinstance(output_file, str):
        f_out = open(output_file, 'w', encoding='utf-8')
        f_callback = f_out.close
    else:
        f_out = output_file
        f_callback = None
    try:
        while True:

            subsequence = text[i:i + MAX_SUBSEQUENCE_LEN]
            subsequence_pauses = pauses[i:i + MAX_SUBSEQUENCE_LEN]

            if not subsequence:
                break

            converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

            y = predict_function(to_array(converted_subsequence), to_array(subsequence_pauses, dtype=theano.config.floatX))

            f_out.write(subsequence[0])

            last_eos_idx = 0
            punctuations = []
            for y_t in y:

                p_i = np.argmax(y_t.flatten())
                punctuation = reverse_punctuation_vocabulary[p_i]

                punctuations.append(punctuation)

                if punctuation in data.EOS_TOKENS:
                    last_eos_idx = len(punctuations) # we intentionally want the index of next element

            if subsequence[-1] == data.END:
                step = len(subsequence) - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subsequence) - 1

            for j in range(step):
                f_out.write(" " + punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
                if j < step - 1:
                    f_out.write(subsequence[1 + j])

            if subsequence[-1] == data.END:
                break

            i += step
    finally:
        if callable(f_callback):
            f_callback()


def restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, predict_function):
    i = 0
    if isinstance(output_file, str):
        f_out = open(output_file, 'w', encoding='utf-8')
        f_callback = f_out.close
    else:
        f_out = output_file
        f_callback = None
    try:
        while True:

            subsequence = text[i:i + MAX_SUBSEQUENCE_LEN]

            if not subsequence:
                break

            converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

            y = predict_function(to_array(converted_subsequence))

            f_out.write(subsequence[0])

            last_eos_idx = 0
            punctuations = []
            for y_t in y:

                p_i = np.argmax(y_t.flatten())
                punctuation = reverse_punctuation_vocabulary[p_i]

                punctuations.append(punctuation)

                if punctuation in data.EOS_TOKENS:
                    last_eos_idx = len(punctuations) # we intentionally want the index of next element

            if subsequence[-1] == data.END:
                step = len(subsequence) - 1
            elif last_eos_idx != 0:
                step = last_eos_idx
            else:
                step = len(subsequence) - 1

            for j in range(step):
                f_out.write(" " + punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
                if j < step - 1:
                    f_out.write(subsequence[1 + j])

            if subsequence[-1] == data.END:
                break

            i += step
    finally:
        if callable(f_callback):
            f_callback()


class Punctuator:

    def model_exists(self, fn):
        if isinstance(fn, bytes):
            return fn
        if os.path.isfile(fn):
            return fn
        _fn = os.path.join(PUNCTUATOR_DATA_DIR, fn)
        if os.path.isfile(_fn):
            return _fn

    def __init__(self, model_file, use_pauses=False):

        model_file = self.model_exists(model_file)
        assert model_file, 'Model %s does not exist.' % model_file

        self.model_file = model_file
        self.use_pauses = use_pauses

        x = T.imatrix('x')

        if use_pauses:

            p = T.matrix('p')

            logging.info("Loading model parameters...")
            if isinstance(model_file, bytes):
                net, _ = models.loads(model_file, 1, x, p)
            else:
                net, _ = models.load(model_file, 1, x, p)
            logging.info("Building model...")
            self.predict = theano.function(inputs=[x, p], outputs=net.y)

        else:

            logging.info("Loading model parameters...")
            if isinstance(model_file, bytes):
                net, _ = models.loads(model_file, 1, x)
            else:
                net, _ = models.load(model_file, 1, x)
            logging.info("Building model...")
            self.predict = theano.function(inputs=[x], outputs=net.y)

        self.net = net
        self.word_vocabulary = net.x_vocabulary
        self.punctuation_vocabulary = net.y_vocabulary

        self.reverse_word_vocabulary = {v: k for k, v in self.word_vocabulary.items()}
        self.reverse_punctuation_vocabulary = {v: k for k, v in self.punctuation_vocabulary.items()}

    def save(self, fn):
        assert isinstance(fn, str)
        with open(fn, 'wb') as fout:
            pickle.dump(self, fout)

    @classmethod
    def load(cls, fn):
        assert isinstance(fn, str)
        with open(fn, 'rb') as fin:
            return pickle.load(fin)

    def punctuate(self, input_text, escape=True):

        text = [
            w for w in input_text.split() if w not in self.punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)
        ] + [data.END]
        pauses = [float(s.replace(data.PAUSE_PREFIX, "").replace(">", "")) for s in input_text.split() if s.startswith(data.PAUSE_PREFIX)]

        fout = StringIO()
        if self.use_pauses:
            if not pauses:
                pauses = [0.0 for _ in range(len(text) - 1)]
            restore_with_pauses(fout, text, pauses, self.word_vocabulary, self.reverse_punctuation_vocabulary, self.predict)
        else:
            restore(fout, text, self.word_vocabulary, self.reverse_punctuation_vocabulary, self.predict)

        # Convert tokenize punctuation to normal punctuation.
        if escape:
            fout2 = StringIO()
            convert(fout.getvalue(), fout2)

        output_text = fout2.getvalue()

        if output_text and not output_text.endswith('.'):
            output_text += '.'

        # Correct "'s" capitalization.
        output_text = re.sub(r"'[a-zA-Z]+\b", lambda m: m.group(0).lower(), output_text)

        # Correct I capitalizations.
        output_text = re.sub(r"\bi'm\b", "I'm", output_text)
        output_text = re.sub(r"\bi've\b", "I've", output_text)
        output_text = re.sub(r"\bi\b", "I", output_text)

        return output_text


def command_line_runner():

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        sys.exit("Model file path argument missing")
    if model_file[0] not in ('.', '..', '/'):
        model_file = os.path.join(PUNCTUATOR_DATA_DIR, model_file)
    assert os.path.isfile(model_file), 'Specified model file does not exist: %s' % model_file

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        sys.exit("Output file path argument missing")

    input_text = open(sys.stdin.fileno(), 'r', encoding='utf-8').read().strip()
    if not input_text:
        sys.exit("Input text from stdin missing.")

    use_pauses = len(sys.argv) > 3 and bool(int(sys.argv[3]))

    p = Punctuator(model_file, use_pauses)
    output_text = p.punctuate(input_text)
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write(output_text)


if __name__ == '__main__':
    command_line_runner()
