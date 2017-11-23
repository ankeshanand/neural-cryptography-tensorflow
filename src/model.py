import tensorflow as tf
import numpy as np

import matplotlib
# OSX fix
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import seaborn as sns

from layers import conv_layer
from config import *
from utils import init_weights, gen_data


class CryptoNet(object):
    def __init__(self, sess, msg_len=MSG_LEN, batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
        """
        Args:
            sess: TensorFlow session
            msg_len: The length of the input message to encrypt.
            key_len: Length of Alice and Bob's private key.
            batch_size: Minibatch size for each adversarial training
            epochs: Number of epochs in the adversarial training
            learning_rate: Learning Rate for Adam Optimizer
        """

        self.sess = sess
        self.msg_len = msg_len
        self.key_len = self.msg_len
        self.N = self.msg_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.build_model()

    def build_model(self):
        # Weights for fully connected layers
        self.w_alice = init_weights("alice_w", [2 * self.N, 2 * self.N])
        self.w_bob = init_weights("bob_w", [2 * self.N, 2 * self.N])
        self.w_eve1 = init_weights("eve_w1", [self.N, 2 * self.N])
        self.w_eve2 = init_weights("eve_w2", [2 * self.N, 2 * self.N])

        # Placeholder variables for Message and Key
        self.msg = tf.placeholder("float", [None, self.msg_len])
        self.key = tf.placeholder("float", [None, self.key_len])

        # Alice's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.alice_input = tf.concat([self.msg, self.key],1)
        self.alice_hidden = tf.nn.sigmoid(tf.matmul(self.alice_input, self.w_alice))
        self.alice_hidden = tf.expand_dims(self.alice_hidden, 2)
        self.alice_output = tf.squeeze(conv_layer(self.alice_hidden, "alice"))

        # Bob's network
        # FC layer -> Conv Layer (4 1-D convolutions)
        self.bob_input = tf.concat([self.alice_output, self.key],1)
        self.bob_hidden = tf.nn.sigmoid(tf.matmul(self.bob_input, self.w_bob))
        self.bob_hidden = tf.expand_dims(self.bob_hidden, 2)
        self.bob_output = tf.squeeze(conv_layer(self.bob_hidden, "bob"))

        # Eve's network
        # FC layer -> FC layer -> Conv Layer (4 1-D convolutions)
        self.eve_input = self.alice_output
        self.eve_hidden1 = tf.nn.sigmoid(tf.matmul(self.eve_input, self.w_eve1))
        self.eve_hidden2 = tf.nn.sigmoid(tf.matmul(self.eve_hidden1, self.w_eve2))
        self.eve_hidden2 = tf.expand_dims(self.eve_hidden2, 2)
        self.eve_output = tf.squeeze(conv_layer(self.eve_hidden2, "eve"))

    def train(self):
        # Loss Functions
        self.decrypt_err_eve = tf.reduce_mean(tf.abs(self.msg - self.eve_output))
        self.decrypt_err_bob = tf.reduce_mean(tf.abs(self.msg - self.bob_output))
        self.loss_bob = self.decrypt_err_bob + (1. - self.decrypt_err_eve) ** 2.

        # Get training variables corresponding to each network
        self.t_vars = tf.trainable_variables()
        self.alice_or_bob_vars = [var for var in self.t_vars if 'alice_' in var.name or 'bob_' in var.name]
        self.eve_vars = [var for var in self.t_vars if 'eve_' in var.name]

        # Build the optimizers
        self.bob_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.loss_bob, var_list=self.alice_or_bob_vars)
        self.eve_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(
            self.decrypt_err_eve, var_list=self.eve_vars)

        self.bob_errors, self.eve_errors = [], []

        # Begin Training
        tf.global_variables_initializer().run()
        for i in range(self.epochs):
            iterations = 2000

            print 'Training Alice and Bob, Epoch:', i + 1
            bob_loss, _ = self._train('bob', iterations)
            self.bob_errors.append(bob_loss)

            print 'Training Eve, Epoch:', i + 1
            _, eve_loss = self._train('eve', iterations)
            self.eve_errors.append(eve_loss)

        self.plot_errors()

    def _train(self, network, iterations):
        bob_decrypt_error, eve_decrypt_error = 1., 1.

        bs = self.batch_size
        # Train Eve for two minibatches to give it a slight computational edge
        if network == 'eve':
            bs *= 2

        for i in range(iterations):
            msg_in_val, key_val = gen_data(n=bs, msg_len=self.msg_len, key_len=self.key_len)

            if network == 'bob':
                _, decrypt_err = self.sess.run([self.bob_optimizer, self.decrypt_err_bob],
                                               feed_dict={self.msg: msg_in_val, self.key: key_val})
                bob_decrypt_error = min(bob_decrypt_error, decrypt_err)

            elif network == 'eve':
                _, decrypt_err = self.sess.run([self.eve_optimizer, self.decrypt_err_eve],
                                               feed_dict={self.msg: msg_in_val, self.key: key_val})
                eve_decrypt_error = min(eve_decrypt_error, decrypt_err)

        return bob_decrypt_error, eve_decrypt_error

    def plot_errors(self):
        """
        Plot Lowest Decryption Errors achieved by Bob and Eve per epoch
        """
        sns.set_style("darkgrid")
        plt.plot(self.bob_errors)
        plt.plot(self.eve_errors)
        plt.legend(['bob', 'eve'])
        plt.xlabel('Epoch')
        plt.ylabel('Lowest Decryption error achieved')
        plt.show()

