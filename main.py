import tensorflow as tf
import RNNCell
import getData
INPUT_SIZE = 27
HIDDEN_SIZE = 256
BATCH_SIZE = 100
FILENAME = 'malenames.txt'
SAVE_PATH = './Models/Word_GRNN/model.ckpt'
RESTORE = False
LEARNING_RATE = 1e-2
ITERATIONS = 10000


def train(iterations, loss, train, y, final_output = None):
    bg = getData.get_batch_generator(FILENAME, BATCH_SIZE, num_outputs)
    for i in range(iterations):
        batch = next(bg, None)
        while batch is None:
            bg = getData.get_batch_generator(FILENAME, 10, 10)
            batch = next(bg, None)
            print('Epoch')
        loss_val, _ = sess.run([loss, train], feed_dict = {y : batch})
        if i % 100 == 0:
            print(loss_val)
            saver.save(sess, SAVE_PATH)
            example = sess.run([final_output])
            getData.print_words(example)


def define_train(num_outputs):
    GRNN = RNNCell.GRNNCell(INPUT_SIZE, HIDDEN_SIZE)
    y = tf.placeholder(tf.uint8,shape=(num_outputs, BATCH_SIZE))
    y_one_hot = tf.one_hot(y, INPUT_SIZE)
    hidden = tf.zeros((BATCH_SIZE, HIDDEN_SIZE))
    x = tf.zeros((BATCH_SIZE, INPUT_SIZE))
    loss= tf.constant(0, dtype = tf.float32)
    for i in range(num_outputs):
        output, hidden = GRNN(x, hidden)
        x = tf.squeeze(tf.gather(y_one_hot, tf.constant(i), axis = 0))
        print(x.get_shape())
        l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x, logits = output))
        loss = loss + l
    loss = loss / num_outputs
    train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    return loss, train, y

def define_output(num_outputs, x = None):
    GRNN = RNNCell.GRNNCell(INPUT_SIZE, HIDDEN_SIZE, reuse = True)
    hidden = tf.zeros((BATCH_SIZE, HIDDEN_SIZE))
    if x is None:
        x = tf.zeros((BATCH_SIZE, INPUT_SIZE))
    loss= tf.constant(0)
    final_output = None
    for i in range(num_outputs):
        output, hidden = GRNN(x, hidden)
        output = tf.squeeze(output)
        softmax = tf.nn.softmax(output, axis = 1)
        print(softmax.get_shape())
        choices = tf.multinomial(output, 1)
        print(choices.get_shape())
        if final_output is None:
            final_output = choices
        else:
            final_output = tf.concat([final_output, choices], axis = 1)
        x = tf.one_hot(tf.squeeze(choices), INPUT_SIZE)
    return final_output

if __name__ == "__main__":
    num_outputs = 10
    sess = tf.Session()
    loss, train_var, y = define_train(num_outputs)
    final_output = define_output(num_outputs)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    if SAVE_PATH is not None and RESTORE:
        saver.restore(sess, SAVE_PATH)
    else:
        saver.save(sess, SAVE_PATH)
    train(ITERATIONS, loss, train_var, y, final_output = final_output)
