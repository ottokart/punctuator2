# coding: utf-8

from __future__ import division, print_function

import models
import data

import theano
import sys
from io import open

import theano.tensor as T
import numpy as np

# pylint: disable=redefined-outer-name


def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T


def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    return punct_token[0]


def punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, text, f_out, show_unk):

    if not text:
        sys.exit("Input text from stdin missing.")

    text = [w for w in text.split() if w not in punctuation_vocabulary] + [data.END]

    i = 0

    while True:

        subsequence = text[i:i + data.MAX_SEQUENCE_LEN]

        if not subsequence:
            break

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]
        if show_unk:
            subsequence = [reverse_word_vocabulary[w] for w in converted_subsequence]

        y = predict(to_array(converted_subsequence))

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


if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        sys.exit("Model file path argument missing")

    show_unk = False
    if len(sys.argv) > 2:
        show_unk = bool(int(sys.argv[2]))

    x = T.imatrix('x')

    print("Loading model parameters...")
    net, _ = models.load(model_file, 1, x)

    print("Building model...")
    predict = theano.function(inputs=[x], outputs=net.y)
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary
    reverse_word_vocabulary = {v: k for k, v in net.x_vocabulary.items()}
    reverse_punctuation_vocabulary = {v: k for k, v in net.y_vocabulary.items()}

    with open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False) as f_out:
        while True:
            try:
                text = raw_input("\nTEXT: ").decode('utf-8')
            except NameError:
                text = input("\nTEXT: ")

            punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, text, f_out, show_unk)
            f_out.flush()
