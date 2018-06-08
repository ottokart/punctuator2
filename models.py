# coding: utf-8
from __future__ import division, print_function

import theano
try:
    import cPickle
    cpickle_options = {}
except ImportError:
    import _pickle as cPickle
    cpickle_options = { 'encoding': 'latin-1' }
import os
import theano.tensor as T
import numpy as np

def PReLU(a, x):
    return T.maximum(0.0, x) + a * T.minimum(0.0, x)

def ReLU(x):
    return T.maximum(0.0, x)

def _get_shape(i, o, keepdims):
    if (i == 1 or o == 1) and not keepdims:
        return (max(i,o),)
    else:
        return (i, o)

def _slice(tensor, size, i):
    """Gets slice of columns of the tensor"""
    if tensor.ndim == 2:
        return tensor[:, i*size:(i+1)*size]
    elif tensor.ndim == 1:
        return tensor[i*size:(i+1)*size]
    else:
        raise NotImplementedError("Tensor should be 1 or 2 dimensional")

def weights_const(i, o, name, const, keepdims=False):
    W_values = np.ones(_get_shape(i, o, keepdims)).astype(theano.config.floatX) * const
    return theano.shared(value=W_values, name=name, borrow=True)

def weights_identity(i, o, name, const, keepdims=False):
    #"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units" (2015) (http://arxiv.org/abs/1504.00941)
    W_values = np.eye(*_get_shape(i, o, keepdims)).astype(theano.config.floatX) * const
    return theano.shared(value=W_values, name=name, borrow=True)

def weights_Glorot(i, o, name, rng, is_logistic_sigmoid=False, keepdims=False):
    #http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    d = np.sqrt(6. / (i + o))
    if is_logistic_sigmoid:
        d *= 4.
    W_values = rng.uniform(low=-d, high=d, size=_get_shape(i, o, keepdims)).astype(theano.config.floatX)
    return theano.shared(value=W_values, name=name, borrow=True)

def load(file_path, minibatch_size, x, p=None):
    import models
    try:
        import cPickle
    except ImportError:
        import _pickle as cPickle
    import theano
    import numpy as np

    with open(file_path, 'rb') as f:
        state = cPickle.load(f, **cpickle_options)

    Model = getattr(models, state["type"])

    rng = np.random
    rng.set_state(state["random_state"])

    net = Model(
        rng=rng,
        x=x,
        minibatch_size=minibatch_size,
        n_hidden=state["n_hidden"],
        x_vocabulary=state["x_vocabulary"],
        y_vocabulary=state["y_vocabulary"],
        stage1_model_file_name=state.get("stage1_model_file_name", None),
        p=p
        )

    for net_param, state_param in zip(net.params, state["params"]):
        net_param.set_value(state_param, borrow=True)

    gsums = [theano.shared(gsum) for gsum in state["gsums"]] if state["gsums"] else None

    return net, (gsums, state["learning_rate"], state["validation_ppl_history"], state["epoch"], rng)

class GRULayer(object):

    def __init__(self, rng, n_in, n_out, minibatch_size):
        super(GRULayer, self).__init__()
        # Notation from: An Empirical Exploration of Recurrent Network Architectures

        self.n_in = n_in
        self.n_out = n_out

        # Initial hidden state
        self.h0 = theano.shared(value=np.zeros((minibatch_size, n_out)).astype(theano.config.floatX), name='h0', borrow=True)

        # Gate parameters:
        self.W_x = weights_Glorot(n_in, n_out*2, 'W_x', rng)
        self.W_h = weights_Glorot(n_out, n_out*2, 'W_h', rng)
        self.b = weights_const(1, n_out*2, 'b', 0)
        # Input parameters
        self.W_x_h = weights_Glorot(n_in, n_out, 'W_x_h', rng)
        self.W_h_h = weights_Glorot(n_out, n_out, 'W_h_h', rng)
        self.b_h = weights_const(1, n_out, 'b_h', 0)

        self.params = [self.W_x, self.W_h, self.b, self.W_x_h, self.W_h_h, self.b_h]

    def step(self, x_t, h_tm1):

        rz = T.nnet.sigmoid(T.dot(x_t, self.W_x) + T.dot(h_tm1, self.W_h) + self.b)
        r = _slice(rz, self.n_out, 0)
        z = _slice(rz, self.n_out, 1)

        h = T.tanh(T.dot(x_t, self.W_x_h) + T.dot(h_tm1 * r, self.W_h_h) + self.b_h)

        h_t = z * h_tm1 + (1. - z) * h

        return h_t

class GRU(object):

    def __init__(self, rng, x, minibatch_size, n_hidden, x_vocabulary, y_vocabulary, stage1_model_file_name=None, p=None):

        assert not stage1_model_file_name and not p, "Stage 1 model can't have stage 1 model"

        x_vocabulary_size = len(x_vocabulary)
        y_vocabulary_size = len(y_vocabulary)

        self.n_hidden = n_hidden
        self.x_vocabulary = x_vocabulary
        self.y_vocabulary = y_vocabulary

        # input model
        pretrained_embs_path = "We.pcl"
        if os.path.exists(pretrained_embs_path):
            print("Found pretrained embeddings in '%s'. Using them..." % pretrained_embs_path)
            with open(pretrained_embs_path, 'rb') as f:
                We = cPickle.load(f, **cpickle_options)
            n_emb = len(We[0])
            We.append([0.1]*n_emb) # END
            We.append([0.0]*n_emb) # UNK - both quite arbitrary initializations

            We = np.array(We).astype(theano.config.floatX)
            self.We = theano.shared(value=We, name="We", borrow=True)
        else:
            n_emb = n_hidden
            self.We = weights_Glorot(x_vocabulary_size, n_emb, 'We', rng) # Share embeddings between forward and backward model

        self.GRU_f = GRULayer(rng=rng, n_in=n_emb, n_out=n_hidden, minibatch_size=minibatch_size)
        self.GRU_b = GRULayer(rng=rng, n_in=n_emb, n_out=n_hidden, minibatch_size=minibatch_size)

        # output model
        self.GRU = GRULayer(rng=rng, n_in=n_hidden*2, n_out=n_hidden, minibatch_size=minibatch_size)
        self.Wy = weights_const(n_hidden, y_vocabulary_size, 'Wy', 0)
        self.by = weights_const(1, y_vocabulary_size, 'by', 0)

        # attention model
        n_attention = n_hidden * 2 # to match concatenated forward and reverse model states
        self.Wa_h = weights_Glorot(n_hidden, n_attention, 'Wa_h', rng) # output model previous hidden state to attention model weights
        self.Wa_c = weights_Glorot(n_attention, n_attention, 'Wa_c', rng) # contexts to attention model weights
        self.ba = weights_const(1, n_attention, 'ba', 0)
        self.Wa_y = weights_Glorot(n_attention, 1, 'Wa_y', rng) # gives weights to contexts

        # Late fusion parameters
        self.Wf_h = weights_const(n_hidden, n_hidden, 'Wf_h', 0)
        self.Wf_c = weights_const(n_attention, n_hidden, 'Wf_c', 0)
        self.Wf_f = weights_const(n_hidden, n_hidden, 'Wf_f', 0)
        self.bf = weights_const(1, n_hidden, 'by', 0)

        self.params = [self.We,
                       self.Wy, self.by,
                       self.Wa_h, self.Wa_c, self.ba, self.Wa_y,
                       self.Wf_h, self.Wf_c, self.Wf_f, self.bf]

        self.params += self.GRU.params + self.GRU_f.params + self.GRU_b.params

        # bi-directional recurrence
        def input_recurrence(x_f_t, x_b_t, h_f_tm1, h_b_tm1):
            h_f_t = self.GRU_f.step(x_t=x_f_t, h_tm1=h_f_tm1)
            h_b_t = self.GRU_b.step(x_t=x_b_t, h_tm1=h_b_tm1)
            return [h_f_t, h_b_t]

        def output_recurrence(x_t, h_tm1, Wa_h, Wa_y, Wf_h, Wf_c, Wf_f, bf, Wy, by, context, projected_context):

            # Attention model
            h_a = T.tanh(projected_context + T.dot(h_tm1, Wa_h))
            alphas = T.exp(T.dot(h_a, Wa_y))
            alphas = alphas.reshape((alphas.shape[0], alphas.shape[1])) # drop 2-axis (sized 1)
            alphas = alphas / alphas.sum(axis=0, keepdims=True)
            weighted_context = (context * alphas[:,:,None]).sum(axis=0)

            h_t = self.GRU.step(x_t=x_t, h_tm1=h_tm1)

            # Late fusion
            lfc = T.dot(weighted_context, Wf_c) # late fused context
            fw = T.nnet.sigmoid(T.dot(lfc, Wf_f) + T.dot(h_t, Wf_h) + bf) # fusion weights
            hf_t = lfc * fw + h_t # weighted fused context + hidden state

            z = T.dot(hf_t, Wy) + by
            y_t = T.nnet.softmax(z)

            return [h_t, hf_t, y_t, alphas]

        x_emb = self.We[x.flatten()].reshape((x.shape[0], minibatch_size, n_emb))

        [h_f_t, h_b_t], _ = theano.scan(fn=input_recurrence,
            sequences=[x_emb, x_emb[::-1]], # forward and backward sequences
            outputs_info=[self.GRU_f.h0, self.GRU_b.h0])

        # 0-axis is time steps, 1-axis is batch size and 2-axis is hidden layer size
        context = T.concatenate([h_f_t, h_b_t[::-1]], axis=2)
        projected_context = T.dot(context, self.Wa_c) + self.ba

        [_, self.last_hidden_states, self.y, self.alphas], _ = theano.scan(fn=output_recurrence,
            sequences=[context[1:]], # ignore the 1st word in context, because there's no punctuation before that
            non_sequences=[self.Wa_h, self.Wa_y, self.Wf_h, self.Wf_c, self.Wf_f, self.bf, self.Wy, self.by, context, projected_context],
            outputs_info=[self.GRU.h0, None, None, None])

        print("Number of parameters is %d" % sum(np.prod(p.shape.eval()) for p in self.params))

        self.L1 = sum(abs(p).sum() for p in self.params)
        self.L2_sqr = sum((p**2).sum() for p in self.params)

    def cost(self, y):
        num_outputs = self.y.shape[0]*self.y.shape[1] # time steps * number of parallel sequences in batch
        output = self.y.reshape((num_outputs, self.y.shape[2]))
        return -T.sum(T.log(output[T.arange(num_outputs), y.flatten()]))

    def save(self, file_path, gsums=None, learning_rate=None, validation_ppl_history=None, best_validation_ppl=None, epoch=None, random_state=None):
        try:
            import cPickle
        except ImportError:
            import _pickle as cPickle
        state = {
            "type":                     self.__class__.__name__,
            "n_hidden":                 self.n_hidden,
            "x_vocabulary":             self.x_vocabulary,
            "y_vocabulary":             self.y_vocabulary,
            "stage1_model_file_name":   self.stage1_model_file_name if hasattr(self, "stage1_model_file_name") else None,
            "params":                   [p.get_value(borrow=True) for p in self.params],
            "gsums":                    [s.get_value(borrow=True) for s in gsums] if gsums else None,
            "learning_rate":            learning_rate,
            "validation_ppl_history":   validation_ppl_history,
            "epoch":                    epoch,
            "random_state":             random_state
        }

        with open(file_path, 'wb') as f:
            cPickle.dump(state, f, protocol=cPickle.HIGHEST_PROTOCOL)


class GRUstage2(GRU):

    def __init__(self, rng, x, minibatch_size, n_hidden, x_vocabulary, y_vocabulary, stage1_model_file_name, p=None):

        y_vocabulary_size = len(y_vocabulary)

        self.stage1_model_file_name = stage1_model_file_name
        self.stage1, _ = load(stage1_model_file_name, minibatch_size, x)

        self.n_hidden = n_hidden
        self.x_vocabulary = x_vocabulary
        self.y_vocabulary = y_vocabulary

        # output model
        self.GRU = GRULayer(rng=rng, n_in=self.stage1.n_hidden + 1, n_out=n_hidden, minibatch_size=minibatch_size)
        self.Wy = weights_const(n_hidden, y_vocabulary_size, 'Wy', 0)
        self.by = weights_const(1, y_vocabulary_size, 'by', 0)

        self.params = [self.Wy, self.by]
        self.params += self.GRU.params

        def recurrence(x_t, p_t, h_tm1, Wy, by):

            h_t = self.GRU.step(x_t=T.concatenate((x_t, p_t.dimshuffle((0, 'x'))), axis=1), h_tm1=h_tm1)

            z = T.dot(h_t, Wy) + by
            y_t = T.nnet.softmax(z)

            return [h_t, y_t]

        [_, self.y], _ = theano.scan(fn=recurrence,
            sequences=[self.stage1.last_hidden_states, p],
            non_sequences=[self.Wy, self.by],
            outputs_info=[self.GRU.h0, None])

        print("Number of parameters is %d" % sum(np.prod(p.shape.eval()) for p in self.params))
        print("Number of parameters with stage1 params is %d" % sum(np.prod(p.shape.eval()) for p in self.params + self.stage1.params))

        self.L1 = sum(abs(p).sum() for p in self.params)
        self.L2_sqr = sum((p**2).sum() for p in self.params)
