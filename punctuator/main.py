# coding: utf-8
"""
Bi-directional RNN with attention
For a sequence of N words, the model makes N punctuation decisions (no punctuation before the first word,
but there's a decision after the last word or before </S>)
"""
from __future__ import division, print_function

import sys
import os.path
from collections import OrderedDict
from time import time

import theano
import theano.tensor as T
import numpy as np

import models
import data

MAX_EPOCHS = 50
MINIBATCH_SIZE = 128
L2_REG = 0.0
CLIPPING_THRESHOLD = 2.0
PATIENCE_EPOCHS = 1


def get_minibatch(file_name, batch_size, shuffle, with_pauses=False):

    dataset = data.load(file_name)

    if shuffle:
        np.random.shuffle(dataset)

    X_batch = []
    Y_batch = []
    if with_pauses:
        P_batch = []

    if len(dataset) < batch_size:
        print(
            "WARNING: Not enough samples in '%s'. Reduce mini-batch size to %d or use a dataset with at least %d words." %
            (file_name, len(dataset), MINIBATCH_SIZE * data.MAX_SEQUENCE_LEN)
        )

    for subsequence in dataset:

        X_batch.append(subsequence[0])
        Y_batch.append(subsequence[1])
        if with_pauses:
            P_batch.append(subsequence[2])

        if len(X_batch) == batch_size:

            # Transpose, because the model assumes the first axis is time
            X = np.array(X_batch, dtype=np.int32).T
            Y = np.array(Y_batch, dtype=np.int32).T
            if with_pauses:
                P = np.array(P_batch, dtype=theano.config.floatX).T

            if with_pauses:
                yield X, Y, P
            else:
                yield X, Y

            X_batch = []
            Y_batch = []
            if with_pauses:
                P_batch = []


if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        sys.exit("'Model name' argument missing!")

    if len(sys.argv) > 2:
        num_hidden = int(sys.argv[2])
    else:
        sys.exit("'Hidden layer size' argument missing!")

    if len(sys.argv) > 3:
        learning_rate = float(sys.argv[3])
    else:
        sys.exit("'Learning rate' argument missing!")

    model_file_name = "Model_%s_h%d_lr%s.pcl" % (model_name, num_hidden, learning_rate)

    print(num_hidden, learning_rate, model_file_name)

    word_vocabulary = data.read_vocabulary(data.WORD_VOCAB_FILE)
    punctuation_vocabulary = data.iterable_to_dict(data.PUNCTUATION_VOCABULARY)

    x = T.imatrix('x')
    y = T.imatrix('y')
    lr = T.scalar('lr')

    continue_with_previous = False
    if os.path.isfile(model_file_name):

        while True:
            resp = input(
                ("Found an existing model with the name %s. Do you want to:\n[c]ontinue training the existing model?\n" \
                "[r]eplace the existing model and train a new one?\n[e]xit?\n>")
                % model_file_name
            )
            resp = resp.lower().strip()
            if resp not in ('c', 'r', 'e'):
                continue
            if resp == 'e':
                sys.exit()
            elif resp == 'c':
                continue_with_previous = True
            break

    if continue_with_previous:
        print("Loading previous model state")

        net, state = models.load(model_file_name, MINIBATCH_SIZE, x)
        gsums, learning_rate, validation_ppl_history, starting_epoch, rng = state
        best_ppl = min(validation_ppl_history)

    else:
        rng = np.random
        rng.seed(1)

        print("Building model...")
        net = models.GRU(rng=rng, x=x, minibatch_size=MINIBATCH_SIZE, n_hidden=num_hidden, x_vocabulary=word_vocabulary, y_vocabulary=punctuation_vocabulary)

        starting_epoch = 0
        best_ppl = np.inf
        validation_ppl_history = []

        gsums = [theano.shared(np.zeros_like(param.get_value(borrow=True))) for param in net.params]

    cost = net.cost(y) + L2_REG * net.L2_sqr

    gparams = T.grad(cost, net.params)
    updates = OrderedDict()

    # Compute norm of gradients
    norm = T.sqrt(T.sum([T.sum(gparam**2) for gparam in gparams]))

    # Adagrad: "Adaptive subgradient methods for online learning and stochastic optimization" (2011)
    for gparam, param, gsum in zip(gparams, net.params, gsums):
        gparam = T.switch(T.ge(norm, CLIPPING_THRESHOLD), gparam / norm * CLIPPING_THRESHOLD, gparam) # Clipping of gradients
        updates[gsum] = gsum + (gparam**2)
        updates[param] = param - lr * (gparam / (T.sqrt(updates[gsum] + 1e-6)))

    train_model = theano.function(inputs=[x, y, lr], outputs=cost, updates=updates)

    validate_model = theano.function(inputs=[x, y], outputs=net.cost(y))

    print("Training...")
    for epoch in range(starting_epoch, MAX_EPOCHS):
        t0 = time()
        total_neg_log_likelihood = 0
        total_num_output_samples = 0
        iteration = 0
        for X, Y in get_minibatch(data.TRAIN_FILE, MINIBATCH_SIZE, shuffle=True):
            total_neg_log_likelihood += train_model(X, Y, learning_rate)
            total_num_output_samples += np.prod(Y.shape)
            iteration += 1
            if iteration % 100 == 0:
                sys.stdout.write(
                    "PPL: %.4f; Speed: %.2f sps\n" %
                    (np.exp(total_neg_log_likelihood / total_num_output_samples), total_num_output_samples / max(time() - t0, 1e-100))
                )
                sys.stdout.flush()
        print("Total number of training labels: %d" % total_num_output_samples)

        total_neg_log_likelihood = 0
        total_num_output_samples = 0
        for X, Y in get_minibatch(data.DEV_FILE, MINIBATCH_SIZE, shuffle=False):
            total_neg_log_likelihood += validate_model(X, Y)
            total_num_output_samples += np.prod(Y.shape)
        print("Total number of validation labels: %d" % total_num_output_samples)

        ppl = np.exp(total_neg_log_likelihood / total_num_output_samples)
        validation_ppl_history.append(ppl)

        print("Validation perplexity is %s" % np.round(ppl, 4))

        if ppl <= best_ppl:
            best_ppl = ppl
            net.save(
                model_file_name,
                gsums=gsums,
                learning_rate=learning_rate,
                validation_ppl_history=validation_ppl_history,
                best_validation_ppl=best_ppl,
                epoch=epoch,
                random_state=rng.get_state()
            )
        elif best_ppl not in validation_ppl_history[-PATIENCE_EPOCHS:]:
            print("Finished!")
            print("Best validation perplexity was %s" % best_ppl)
            break
