# coding: utf-8

from __future__ import division

import models
import data

import theano
import sys
import codecs

import theano.tensor as T
import numpy as np

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]

def punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, text, f_out, show_unk):

    if len(text) == 0:
        sys.exit("Input text from stdin missing.")

    text = [w for w in text.split() if w not in punctuation_vocabulary] + [data.END]

    i = 0

    while True:

        subsequence = text[i:i+data.MAX_SEQUENCE_LEN]

        if len(subsequence) == 0:
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
                f_out.write(subsequence[1+j])

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

    print "Loading model parameters..."
    net, _ = models.load(model_file, 1, x)

    print "Building model..."
    predict = theano.function(inputs=[x], outputs=net.y)
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary
    reverse_word_vocabulary = {v:k for k,v in net.x_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in net.y_vocabulary.items()}

    with codecs.getwriter('utf-8')(sys.stdout) as f_out:
        while True:
            text = raw_input("\nTEXT: ").decode('utf-8')
            punctuate(predict, word_vocabulary, punctuation_vocabulary, reverse_punctuation_vocabulary, reverse_word_vocabulary, text, f_out, show_unk)
