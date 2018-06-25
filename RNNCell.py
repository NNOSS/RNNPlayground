
#Author: Nick Steelman
#Date: 6/25/18
#gns126@gmail.com
#cleanestmink.com

import tensorflow as tf


def GRNNCell(input_size, hidden_size):
    wz = tf.get_variable('wz', shape = (input_size + hidden_size, hidden_size),
            dtype=tf.float32, initializer = tf.random_uniform_initializer)
    wr = tf.get_variable('wr', shape = (input_size + hidden_size, hidden_size),
            dtype=tf.float32, initializer = tf.random_uniform_initializer)
    bz = tf.get_variable('bz', shape = (hidden_size),
            dtype=tf.float32, initializer = tf.random_uniform_initializer)
    br = tf.get_variable('br', shape = (hidden_size),
            dtype=tf.float32, initializer = tf.random_uniform_initializer)


    w_hat = tf.get_variable('w_hat', shape = (input_size, hidden_size),
            dtype=tf.float32, initializer = tf.random_uniform_initializer)
    wh_hat = tf.get_variable('wh_hat', shape = (hidden_size, hidden_size),
            dtype=tf.float32, initializer = tf.random_uniform_initializer)
    b0 = tf.get_variable('b0', shape = (hidden_size),
            dtype=tf.float32, initializer = tf.zeros_initializer)

    w1 = tf.get_variable('w1', shape = (hidden_size ,input_size),
            dtype=tf.float32, initializer = tf.random_uniform_initializer)
    b1 = tf.get_variable('b1', shape = (input_size),
            dtype=tf.float32, initializer = tf.zeros_initializer)

    def return_output(input_matix, prev_hidden):
        concat = tf.concat([input_matix, prev_hidden], axis = 1)
        z_gate = tf.nn.sigmoid(tf.matmul(concat, wz) + bz)
        r_gate = tf.nn.sigmoid(tf.matmul(concat, wr) + br)
        h_hat = tf.nn.tanh(tf.multiply(r_gate, tf.matmul(prev_hidden, wh_hat)) +
                tf.matmul(input_matrix, w_hat) + b0)
        hidden = tf.multiply(z_gate,  prev_hidden)+ tf.multiply(1-z_gate, h_hat)
        output_matrix = tf.nn.relu(tf.matmul(hidden, w1) + b1)

        return output_matrix, hidden

    return return_output
