from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, psutil
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Imports
import numpy as np
import tensorflow as tf


tf.reset_default_graph()

# Load data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
# other datasets: 'iris', 'boston', 'dbpedia'
train_set = mnist.train.images
train_labels = np.asanyarray(mnist.train.labels, dtype = np.int32)
eval_set = mnist.test.images
eval_labels = np.asanyarray(mnist.test.labels, dtype=np.int32)


# Input layer
X = tf.placeholder(tf.float32, shape = [None, 784])
y = tf.placeholder(tf.int64, shape = [None])

input_layer = tf.reshape(X, [-1,28,28,1])
onehot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=10)

# Convolutional layer #1
conv1 = tf.layers.conv2d(
    inputs = input_layer,
    filters = 32,
    kernel_size = [5, 5],
    padding = 'same',
    activation=tf.nn.relu)

# Comments
# The padding is set to 'SAME' which means the input image
# is padded with zeroes so the size of the output is the same.


# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size= [2, 2], strides = 2)

# Convolutional Layer #2 and Pooling Layer #2
conv2 = tf.layers.conv2d(
    inputs = pool1,
    filters= 64,
    kernel_size = [5, 5],
    padding= 'same',
    activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size= [2, 2], strides = 2)


# Dense layer or fully connected layer
flattened_layer = tf.contrib.layers.flatten(inputs = pool2)

dense = tf.layers.dense(inputs=flattened_layer, units = 1024, activation=tf.nn.relu)
# Or
# dense = tf.contrib.layers.fully_connected(inputs = flattened_layer, units = 1024)

dropout = tf.nn.dropout(dense, keep_prob= 0.5)

# logits layer
logits = tf.layers.dense(inputs = dropout, units = 10)

predictions = {"classes": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits)}

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step()) #tf.train.get_global_step()

correct_prediction = tf.equal(y, tf.argmax(input=logits, axis=1))
accuracy = 100*tf.reduce_mean(tf.to_float(correct_prediction))

global_step_tensor = tf.Variable(0, name="global_step", trainable=False)


# Create experiment directory
experiment_dir = os.path.abspath('./experiments/mnist_CNN')

#checkpoint_dir = '/home/lihuang/Desktop/tfgraph/mnist_cnn/checkpoint_dir/'
checkpoint_dir = os.path.join(experiment_dir, 'checkpoint')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'model')


# Create a saver object and checkpoint directory
saver = tf.train.Saver()
# Load the latest checkpoint
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

# Create directory to write summaries to disk
log_path = os.path.join(experiment_dir, 'summary')
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer = tf.summary.FileWriter(log_path)

# Add summaries to Tensorboard
tf.summary.scalar('loss', loss)
tf.summary.histogram('logits', logits)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('global_step_tensor', global_step_tensor)
summary_op = tf.summary.merge_all()

current_process = psutil.Process()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if latest_checkpoint:
        print('Loading the latest checkpoint from ... \n {}'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)


    for epoch in range(10):
        system_summary = tf.Summary()
        mini_batch = mnist.train.next_batch(100)
        current_loss, _, summary = sess.run([loss, train_op, summary_op],feed_dict = {X: mini_batch[0], y: mini_batch[1] } )
        print("\rTraining step {} of 100, loss= {}".format(epoch+1, current_loss), end="")
        system_summary.value.add(simple_value = epoch, tag = 'epoch')
        system_summary.value.add(simple_value = current_process.cpu_percent(), tag = 'system/cpu_usage')
        system_summary.value.add(simple_value = current_process.memory_percent(memtype='vms'), tag='system/RAM_usage')

        # Write summary to file
        writer.add_summary(summary, global_step=tf.train.global_step(sess, global_step_tensor))
        writer.add_summary(system_summary, epoch)
        writer.flush()
    print("\nTest set accuracy: {}%".format(sess.run(accuracy, feed_dict = {X: eval_set, y: eval_labels})), end="")
    saver.save(# Or tf.get_default_session()
                sess,
                checkpoint_path,
                global_step=tf.train.global_step(sess, global_step_tensor))

writer.close()
# tensorboard --logdir='/home/lihuang/PycharmProjects/TensorFlowTest/experiments/mnist_CNN/summary'  --port 6006

