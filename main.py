import tensorflow as tf
import RNNCell


def train(output):



def define_train(num_outputs):
    GRNN = RNNCell.GRNNCell(INPUT_SIZE, HIDDEN_SIZE)
    y = tf.placeholder(shape=(num_outputs, BATCH_SIZE, INPUT_SIZE))
    hidden = tf.zeros(BATCH_SIZE, HIDDEN_SIZE)
    x = tf.zeros(BATCH_SIZE, INPUT_SIZE)
    loss= tf.constant(0)
    for i in range(num_outputs):
        output, hidden = GRNN(x, hidden)
        x = tf.gather(y, tf.constant(i), axis = 0)
        loss = loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x, logits = output))
    loss = loss / num_outputs
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
