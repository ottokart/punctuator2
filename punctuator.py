# coding: utf-8
from __future__ import division

import models
import data

import theano
import sys
import codecs

import theano.tensor as T
import numpy as np

MAX_SUBSEQUENCE_LEN = 200

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]

def restore_with_pauses(output_file, text, pauses, word_vocabulary, reverse_punctuation_vocabulary, predict_function):
    i = 0
    with codecs.open(output_file, 'w', 'utf-8') as f_out:
        while True:

            subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]
            subsequence_pauses = pauses[i:i+MAX_SUBSEQUENCE_LEN]

            if len(subsequence) == 0:
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
                    f_out.write(subsequence[1+j])

            if subsequence[-1] == data.END:
                break

            i += step

def restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, predict_function):
    i = 0
    with codecs.open(output_file, 'w', 'utf-8') as f_out:
        while True:

            subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

            if len(subsequence) == 0:
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
                    f_out.write(subsequence[1+j])

            if subsequence[-1] == data.END:
                break

            i += step

if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        sys.exit("Model file path argument missing")

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        sys.exit("Output file path argument missing")

    use_pauses = len(sys.argv) > 3 and bool(int(sys.argv[3]))

    x = T.imatrix('x')
    
    if use_pauses:
    
        p = T.matrix('p')

        print "Loading model parameters..."
        net, _ = models.load(model_file, 1, x, p)

        print "Building model..."
        predict = theano.function(
            inputs=[x, p],
            outputs=net.y
        )

    else:

        print "Loading model parameters..."
        net, _ = models.load(model_file, 1, x)

        print "Building model..."
        predict = theano.function(
            inputs=[x],
            outputs=net.y
        )

    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}

    input_text = codecs.getreader('utf-8')(sys.stdin).read()

    if len(input_text) == 0:
        sys.exit("Input text from stdin missing.")

    text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
    pauses = [float(s.replace(data.PAUSE_PREFIX,"").replace(">","")) for s in input_text.split() if s.startswith(data.PAUSE_PREFIX)]

    if not use_pauses or len(pauses) == 0:
        restore(output_file, text, word_vocabulary, reverse_punctuation_vocabulary, predict)
    else:
        restore_with_pauses(output_file, text, pauses, word_vocabulary, reverse_punctuation_vocabulary, predict)