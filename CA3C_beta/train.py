#! /usr/bin/env python

import unittest
from RGBEnv_v1 import MazeEnv
import sys, time
import os
import numpy as np
import tensorflow as tf
import itertools
import shutil
import threading
import multiprocessing
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))
experiment_dir = os.path.abspath('./experiments/')

if import_path not in sys.path:
  sys.path.append(import_path)


from estimators import cnn_lstm
from worker import Worker

start_time = time.time()
# An example of tf.flags
# https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/BEGAN/src/model/flags.py

tf.flags.DEFINE_string("model_dir", experiment_dir, "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", "Breakout-v0", "Name of gym Atari environment, e.g. Breakout-v0")
tf.flags.DEFINE_integer("t_max", 20, "Number of steps before performing an update")
tf.flags.DEFINE_integer("max_global_steps",None, "Stop training after this many steps in the environment. Defaults to running indefinitely.")
tf.flags.DEFINE_integer("eval_every", 300, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", False, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("parallelism", 40, "Number of threads to run. If not set we run [num_cpu_cores] threads.")

FLAGS = tf.flags.FLAGS

def make_env():
  # env = gym.envs.make(FLAGS.env)
  env = MazeEnv()
  # remove the timelimitwrapper
  # env = env.env
  # if wrap:
  #   env = atari_helpers.AtariEnvWrapper(env)
  return env

# Depending on the game we may have a limited action space
# env_ = make_env()
# if FLAGS.env == "Pong-v0" or FLAGS.env == "Breakout-v0":
#   VALID_ACTIONS = list(range(4))
# else:
#   VALID_ACTIONS = list(range(env_.action_space.n))
# env_.close()

VALID_ACTIONS = [0, 1, 2, 3]


# Set the number of workers
NUM_WORKERS = multiprocessing.cpu_count()
if FLAGS.parallelism:
  NUM_WORKERS = FLAGS.parallelism

MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model')

# Optionally empty model directory
if FLAGS.reset:
  shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
  os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

with tf.device("/cpu:0"):

  # Keeps track of the number of updates we've performed
  global_step = tf.Variable(0, name="global_step", trainable=False)

  # Global policy and value netsValueError: No gradients provided for any variable
  with tf.variable_scope("global") as vs:
    global_net = cnn_lstm(feature_space=256, action_space=4, reuse=True)

  # Global step iterator
  global_counter = itertools.count()
  # saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.05, max_to_keep=10)
  saver = tf.train.Saver(max_to_keep=10)
  # Create worker graphs
  workers = []
  for worker_id in range(NUM_WORKERS):
    # We only write summaries in one of the workers because they're
    # pretty much identical and writing them on all workers
    # would be a waste of space
    worker_summary_writer = None
    if worker_id == 0:
      worker_summary_writer = summary_writer

    worker = Worker(
      name="worker_{}".format(worker_id),
      start_time=start_time,
      saver=saver,
      checkpoint_path=checkpoint_path,
      env=make_env(),
      global_net=global_net,
      global_counter=global_counter,
      discount_factor = 0.99,
      summary_writer=worker_summary_writer,
      max_global_steps=FLAGS.max_global_steps)
    workers.append(worker)


  # Used to occasionally save videos for our policy net
  # and write episode rewards to Tensorboard
  # pe = PolicyMonitor(
  #   env=make_env(),
  #   policy_net=policy_net,
  #   summary_writer=summary_writer,
  #   saver=saver)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # https://www.tensorflow.org/api_docs/python/tf/train/Coordinator
  coord = tf.train.Coordinator()

  # Load a previous checkpoint if it exists
  latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
  if latest_checkpoint:
    print("Loading model checkpoint: {}".format(latest_checkpoint))
    saver.restore(sess, latest_checkpoint)

  # Start worker threads
  worker_threads = []
  for worker in workers:
    worker_fn = lambda worker=worker: worker.run(sess, coord, FLAGS.t_max)

    t = threading.Thread(target=worker_fn)
    # multithreading example:https://pymotw.com/2/threading/

    t.start()
    worker_threads.append(t)

  # Start a thread for policy eval task
  # monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
  # monitor_thread.start()

  # Wait for all workers to finish
  coord.join(worker_threads)
