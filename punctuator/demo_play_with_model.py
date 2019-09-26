# coding: utf-8

from __future__ import division, print_function

from nltk.tokenize import word_tokenize

import models
import data

import theano
import sys
import re
from io import open

import theano.tensor as T
import numpy as np

# pylint: disable=redefined-outer-name

numbers = re.compile(r'\d')
is_number = lambda x: len(numbers.sub('', x)) / len(x) < 0.6


def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T


def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return ' '
    if punct_token.startswith('-'):
        return ' ' + punct_token[0] + ' '
    return punct_token[0] + ' '


def punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, f_out, show_unk):

    if not words:
        sys.exit("Input text from stdin missing.")

    if words[-1] != data.END:
        words += [data.END]

    i = 0

    while True:

        subsequence = words[i:i + data.MAX_SEQUENCE_LEN]

        if not subsequence:
            break

        converted_subsequence = [word_vocabulary.get("<NUM>" if is_number(w) else w.lower(), word_vocabulary[data.UNK]) for w in subsequence]

        if show_unk:
            subsequence = [reverse_word_vocabulary[w] for w in converted_subsequence]

        y = predict(to_array(converted_subsequence))

        f_out.write(subsequence[0].title())

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
            current_punctuation = punctuations[j]
            f_out.write(convert_punctuation_to_readable(current_punctuation))
            if j < step - 1:
                if current_punctuation in data.EOS_TOKENS:
                    f_out.write(subsequence[1 + j].title())
                else:
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

    human_readable_punctuation_vocabulary = [p[0] for p in punctuation_vocabulary if p != data.SPACE]
    tokenizer = word_tokenize
    untokenizer = lambda text: text.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")

    with open(sys.stdout.fileno(), 'w', encoding='utf-8', closefd=False) as f_out:
        while True:
            try:
                text = raw_input("\nTEXT: ").decode('utf-8')
            except NameError:
                text = input("\nTEXT: ")

            words = [
                w for w in untokenizer(' '.join(tokenizer(text))).split() if w not in punctuation_vocabulary and w not in human_readable_punctuation_vocabulary
            ]

            punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, words, f_out, show_unk)
            f_out.flush()
