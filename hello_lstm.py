# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss

import numpy as np

X_train = "Hello, World!"
# X_train = "안녕하세요, World!"

batch_size = 1


# ==============================================================================
def create_vocabulary(sequence):
    vocab = {}
    for i in range(len(sequence)):
        ch = X_train[i]
        if ch in vocab:
            vocab[ch] += 1
        else:
            vocab[ch] = 1
    vocab_rev = sorted(vocab, key=vocab.get, reverse=True)
    vocab = dict([(x, y) for (y, x) in enumerate(vocab_rev)])
    return vocab, vocab_rev


def sentence_to_token_ids(sentence, vocabulary):
    characters = [sentence[i:i+1] for i in range(0, len(sentence), 1)]
    return [vocabulary.get(w) for w in characters]


def token_ids_to_one_hot(token_ids, num_classes=10):
    token_ids_one_hot = np.zeros((len(token_ids), num_classes))
    token_ids_one_hot[np.arange(len(token_ids)), token_ids] = 1
    return token_ids_one_hot


# ==============================================================================
sequence_length = len(X_train) - 1

X_train_vocab, X_train_vocab_rev = create_vocabulary(X_train)
hidden_size = len(X_train_vocab)
num_classes = len(X_train_vocab)

X_train_ids = sentence_to_token_ids(X_train, X_train_vocab)
X_data = X_train_ids[:-1]
Y_data = X_train_ids[1:]
X_data_one_hot = [token_ids_to_one_hot(X_data, num_classes)]
Y_data = [Y_data]


# ==============================================================================
X = tf.placeholder(tf.float32, [None, sequence_length, hidden_size])
Y = tf.placeholder(tf.int32, [None, sequence_length])


cell = BasicLSTMCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell,
                                     X,
                                     initial_state=initial_state,
                                     dtype=tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = fully_connected(inputs=X_for_fc,
                          num_outputs=num_classes,
                          activation_fn=None)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = sequence_loss(logits=outputs,
                              targets=Y,
                              weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(25):
        l, _ = sess.run([loss, train], feed_dict={X: X_data_one_hot, Y: Y_data})
        result = sess.run(prediction, feed_dict={X: X_data_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", Y_data)
        result_str = [X_train_vocab_rev[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
