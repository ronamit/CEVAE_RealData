import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import sys
from tensorflow.contrib.layers.python.layers import initializers


def fc_net(inp, layers, out_layers, scope, lamba=1e-3, activation=tf.nn.relu, reuse=None,
           weights_initializer=initializers.xavier_initializer(uniform=False)):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=activation,
                        normalizer_fn=None,
                        weights_initializer=weights_initializer,
                        reuse=reuse,
                        weights_regularizer=slim.l2_regularizer(lamba)):

        if layers:
            h = slim.stack(inp, slim.fully_connected, layers, scope=scope)
            if not out_layers:
                return h
        else:
            h = inp
        outputs = []
        for i, (outdim, activation) in enumerate(out_layers):
            o1 = slim.fully_connected(h, outdim, activation_fn=activation, scope=scope + '_{}'.format(i + 1))
            outputs.append(o1)
        return outputs if len(outputs) > 1 else outputs[0]


def get_y0_y1(sess, y, f0, f1, shape=(), L=1, verbose=True):
    y0, y1 = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    ymean = y.mean()
    for l in range(L):
        if L > 1 and verbose:
            sys.stdout.write('\r Sample {}/{}'.format(l + 1, L))
            sys.stdout.flush()
        y0 += sess.run(ymean, feed_dict=f0) / L
        y1 += sess.run(ymean, feed_dict=f1) / L

    if L > 1 and verbose:
        print
    return y0, y1




def matching_estimate(feat_train, t_train, y_train, feat_test, n_neighbours=5):
    n_test = feat_test.shape[0]
    n_train =  feat_train.shape[0]
    est_y0 = np.zeros((n_test, 1))
    est_y1 = np.zeros((n_test, 1))

    # penalty vectors:
    train_idx = np.arange(n_train)
    t_train = t_train.flatten()
    train_idx0 = train_idx[t_train == 0]
    feat_train0 = feat_train[t_train == 0]
    train_idx1 = train_idx[t_train == 1]
    feat_train1 = feat_train[t_train == 1]

    # For each test sample
    for i_sample, feat_sample in enumerate(feat_test):
        # Find closest train samples with t=0
        dists = np.linalg.norm(feat_sample - feat_train0, axis=1)
        closests_idx = np.argsort(dists)[:n_neighbours]
        closests_idx = train_idx0[closests_idx]
        est_y0[i_sample] = y_train[closests_idx].mean()

        # Find closest train samples with t=1
        dists = np.linalg.norm(feat_sample - feat_train1, axis=1)
        closests_idx = np.argsort(dists)[:n_neighbours]
        closests_idx = train_idx1[closests_idx]
        est_y1[i_sample] = y_train[closests_idx].mean()



    return est_y0, est_y1