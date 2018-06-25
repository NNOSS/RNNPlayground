import tensorflow as tf
import RNNCell
import getData
INPUT_SIZE = 27
HIDDEN_SIZE = 256
BATCH_SIZE = 100
FILENAME = 'malenames.txt'

RESTORE = False


def train(iterations, loss, train, y):
    bg = getData.get_batch_generator(FILENAME, BATCH_SIZE, num_outputs)
    for i in range(iterations):
        batch = next(bg, None)
        while batch is not None:
            bg = getData.get_batch_generator(FILENAME, 10, 10)
            batch = next(bg, None)
        loss, _ = sess.run([loss, train], feed_dict = {y : batch})
        if i % 100 == 0:
            print(loss)


def define_train(num_outputs):
    GRNN = RNNCell.GRNNCell(INPUT_SIZE, HIDDEN_SIZE)
    y = tf.placeholder(shape=(num_outputs, BATCH_SIZE, 1))
    y_one_hot = tf.one_hot(y, INPUT_SIZE, axis = 2)
    hidden = tf.zeros(BATCH_SIZE, HIDDEN_SIZE)
    x = tf.zeros(BATCH_SIZE, INPUT_SIZE)
    loss= tf.constant(0)
    for i in range(num_outputs):
        output, hidden = GRNN(x, hidden)
        x = tf.gather(y_one_hot, tf.constant(i), axis = 0)
        loss = loss + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x, logits = output))
    loss = loss / num_outputs
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    return loss, train, y

def define_output(num_outputs, x = None):
    GRNN = RNNCell.GRNNCell(INPUT_SIZE, HIDDEN_SIZE)
    hidden = tf.zeros(BATCH_SIZE, HIDDEN_SIZE)
    if x is None:
        x = tf.zeros(BATCH_SIZE, INPUT_SIZE)
    loss= tf.constant(0)
    final_output = None
    for i in range(num_outputs):
        output, hidden = GRNN(x, hidden)
        softmax = tf.nn.softmax(output, axis = 1)
        choices = tf.multinomial(softmax, 1)
        if final_output is None:
            final_output = choices
        else:
            final_output = tf.concat([final_output, choices], axis = 1)
        x = tf.one_hot(choices, INPUT_SIZE)
    return final_output

if __name__ == "__main__":
    num_outputs = 10
    sess = tf.session()
    saver = tf.train.Saver()
    loss, train, y = define_train(num_outputs)

    if SAVE_PATH is not None and RESTORE:
        saver.restore(sess, SAVE_PATH)
    else:
        saver.save(sess, SAVE_PATH)

    train(10000, loss, train, y)
