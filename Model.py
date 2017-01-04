#! -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf


def zero_one(target):
    return 0 if target < 0.5 else 1

def _get_weights(shape, _stddev=1.0):
    initial = tf.truncated_normal(shape, stddev=_stddev)
    return tf.Variable(initial)

def _get_biases(shape, value=0.0):
    initial = tf.constant(value, shape=shape)
    return tf.Variable(initial)

class Encoder(object):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        
    def set_model(self, x):
        hidden_dim = self.hidden_dim
        
        # -- first layer ------------
        weight_en1 = _get_weights(shape=[28 * 28 * 1, hidden_dim], _stddev = 0.1)
        bias_en1 = _get_biases([hidden_dim], value=0.0)
        en_h1 = tf.nn.relu(tf.matmul(x, weight_en1) + bias_en1)

        # -- second layer ---------
        weight_en2 = _get_weights(shape=[hidden_dim, hidden_dim], _stddev = 0.1)
        bias_en2 = _get_biases([hidden_dim], value=0.0)
        self.en_h2 = tf.nn.relu(tf.matmul(en_h1, weight_en2) + bias_en2)

        return self.en_h2

class Decoder(object):
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        
    def set_model(self, z, z_dim):
        hidden_dim = self.hidden_dim
        
        weight_fc1 = _get_weights(shape=[z_dim, hidden_dim], _stddev=0.1)
        bias_fc1 = _get_biases([hidden_dim], value=0.0)
        fc1 = tf.nn.relu(tf.matmul(z, weight_fc1) + bias_fc1)

        weight_fc2 = _get_weights(shape=[hidden_dim, hidden_dim], _stddev=0.1)
        bias_fc2 = _get_biases([hidden_dim], value=0.0)
        fc2 = tf.matmul(fc1, weight_fc2) + bias_fc2
        
        weight_fc3 = _get_weights(shape=[hidden_dim, 28 * 28 * 1], _stddev=0.1)
        bias_fc3 = _get_biases([28 * 28 * 1], value=0.0)
        self.fc3 = tf.nn.sigmoid(tf.matmul(fc2, weight_fc3) + bias_fc3)

        return self.fc3
        
class Model(object):
    def __init__(self, _z_dim, _batch_size):
        self.z_dim = _z_dim
        self.batch_size = _batch_size
        
    def set_model(self):
        z_dim = self.z_dim
        hidden_dim = 200

        # input
        self.x = tf.placeholder("float", shape=[None, 28 * 28 * 1])


        encoder = Encoder(hidden_dim)
        decoder = Decoder(hidden_dim)
        
        # set encoder
        en_h2 = encoder.set_model(self.x)

        # calculate z
        weight_mu = _get_weights(shape=[hidden_dim, z_dim], _stddev=0.1)
        bias_mu = _get_biases([z_dim], value=0.0)
        self.mu = tf.matmul(en_h2, weight_mu) + bias_mu

        weight_sigma = _get_weights(shape=[hidden_dim, z_dim], _stddev=0.1)
        bias_sigma = _get_biases([z_dim], value=0.0)
        sigma = tf.exp(tf.matmul(en_h2, weight_sigma) + bias_sigma)

        self.eps = tf.placeholder("float", shape = [None, self.z_dim])
        #eps = tf.random_normal((self.batch_size, z_dim), 0, 1, dtype=tf.float32)
        self.z = self.mu + sigma * self.eps

        # set decoder
        fc3 = decoder.set_model(self.z, z_dim)
        self.average = tf.reshape(fc3, [-1, 28, 28, 1])
        
        # set obj
        obj = -tf.reduce_sum(self.x * tf.log(1.0e-10 + fc3) + (1 - self.x) * tf.log(1.0e-10 + 1 - fc3), 1)
        obj += tf.reduce_sum(self.mu * self.mu/2.0 - tf.log(sigma) + sigma * sigma/2.0, 1)
        obj = tf.reduce_mean(obj)

        # set optimizer
        optimizer = tf.train.AdamOptimizer(0.0001)
        self.train = optimizer.minimize(obj)

    def training(self, sess, inputs):
        sess.run(self.train,
                 feed_dict = {self.x: inputs,
                              self.eps: np.random.randn(self.batch_size, self.z_dim)})

    def encode(self, sess, x):
        return sess.run(self.z,
                        feed_dict = {self.x: x,
                                     self.eps: [[0] * self.z_dim]})
    def generate(self, sess, z):
        tmp, hoge = sess.run([self.average, self.z], feed_dict = {self.z:z})
        ret = []
        for tmp2 in tmp[0]:
            ret.append([zero_one(_[0]) for _ in tmp2])
        return ret
    
    def auto_encode(self, sess, x):
        tmp = sess.run(self.average, feed_dict={self.x: x, self.eps:[[0] * self.z_dim]})
        ret = []
        for tmp2 in tmp[0]:
            ret.append([zero_one(_[0]) for _ in tmp2])
        return ret
    

if __name__ == u'__main__':

    model = Model(10)
    obj = model.set_model()
