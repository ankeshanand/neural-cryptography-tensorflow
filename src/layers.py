import tensorflow as tf
from config import FILTERS


def conv1d(input_, filter_shape, stride, name="conv1d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', shape=filter_shape,
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.conv1d(input_, w, stride, padding='SAME')

        return conv


def conv_layer(hidden_layer_output, name):
    h0 = tf.nn.relu(conv1d(hidden_layer_output, FILTERS[0], stride=1, name=name+'_h0_conv'))
    h1 = tf.nn.relu(conv1d(h0, FILTERS[1], stride=2, name=name+'_h1_conv'))
    h2 = tf.nn.relu(conv1d(h1, FILTERS[2], stride=1, name=name+'_h2_conv'))
    h3 = tf.nn.tanh(conv1d(h2, FILTERS[3], stride=1, name=name+'_h3_conv'))

    return h3
