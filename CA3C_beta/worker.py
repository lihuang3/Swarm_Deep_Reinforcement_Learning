import gym
import sys, time
import os
import itertools
import collections
import numpy as np
import tensorflow as tf

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
  sys.path.append(import_path)

# from lib import plotting
from lib.atari import helpers as atari_helpers
from estimators import ValueEstimator, PolicyEstimator

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def make_copy_params_op(v1_list, v2_list):
  """
  Creates an operation that copies parameters from variable in v1_list to variables in v2_list.
  The ordering of the variables in the lists must be identical.
  """
  v1_list = list(sorted(v1_list, key=lambda v: v.name))
  v2_list = list(sorted(v2_list, key=lambda v: v.name))

  update_ops = []
  for v1, v2 in zip(v1_list, v2_list):
    op = v2.assign(v1)
    update_ops.append(op)

  return update_ops
#>>> ToDo
def make_train_op(local_estimator, global_estimator):
  """
  Creates an op that applies local estimator gradients
  to the global estimator.
  """
  local_grads, _ = zip(*local_estimator.grads_and_vars)
  # Clip gradients
  local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
  _, global_vars = zip(*global_estimator.grads_and_vars)
  local_global_grads_and_vars = list(zip(local_grads, global_vars))
  return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
          global_step=tf.train.get_global_step())


class Worker(object):
  """
  An A3C worker thread. Runs episodes locally and updates global shared value and policy nets.

  Args:
    name: A unique name for this worker
    env: The Gym environment used by this worker
    policy_net: Instance of the globally shared policy net
    value_net: Instance of the globally shared value net
    global_counter: Iterator that holds the global step
    discount_factor: Reward discount factor
    summary_writer: A tf.train.SummaryWriter for Tensorboard summaries
    max_global_steps: If set, stop coordinator when global_counter > max_global_steps
  """
  def __init__(self, name, start_time, saver, checkpoint_path, env, policy_net, value_net, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
    self.name = name
    self.start_time = start_time
    self.saver = saver
    self.checkpoint_path = checkpoint_path
    self.discount_factor = discount_factor
    self.max_global_steps = max_global_steps
    self.global_step = tf.train.get_global_step()
    self.global_policy_net = policy_net
    self.global_value_net = value_net
    self.global_counter = global_counter
    self.local_counter = itertools.count()
    self.summary_writer = summary_writer
    self.env = env
    self.episode = 1
    self.episode_local_step = 0
    self.display_flag = False
    # Create local policy/value nets that are not updated asynchronously
    with tf.variable_scope(name):
      self.policy_net = PolicyEstimator(policy_net.num_outputs)
      self.value_net = ValueEstimator(reuse=True)

    # Op to copy params from global policy/valuenets
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
    self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

    self.state = None

  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # Initial state
      self.state = self.env.reset()
      self.state = np.stack([self.state] * 4, axis=2)
      # Init LSTMs state
      lstm_c_init = np.zeros((1, 256), np.float32)
      lstm_h_init = np.zeros((1, 256), np.float32)
      self.lstm_state = [lstm_c_init, lstm_h_init]
      try:
        while not coord.should_stop():
          # Copy Parameters from the global networks
          sess.run(self.copy_params_op)

          # Collect some experience
          transitions, local_t, global_t = self.run_n_steps(t_max, sess)

          if self.max_global_steps is not None and global_t >= self.max_global_steps:
            tf.logging.info("Reached global step {}. Stopping.".format(global_t))
            coord.request_stop()
            return

          # Update the global networks
          pnet_loss, vnet_loss, _, _=self.update(transitions, sess)
          if self.display_flag:
            training_time = int(time.time()-self.start_time)
            print("Training time: {} d {} hr {} min, global step = {}, {}, Episode = {}, pnet_loss = {:.4E}, vnet_loss = {:.4E}".format(training_time/86400,
                                    (training_time/3600)%24, (training_time/60)%60, global_t, self.name, self.episode, pnet_loss, vnet_loss))
            # self.saver.save(sess, self.checkpoint_path)
            self.display_flag = False

      except tf.errors.CancelledError:
        return

  def _policy_net_predict(self, sess, state, c_in, h_in):
    feed_dict = { self.policy_net.states: [state], self.policy_net.c_in: c_in, self.policy_net.h_in: h_in}
    preds = sess.run(self.policy_net.predictions, feed_dict)
    return preds["probs"][0], preds["features"]

  def _value_net_predict(self, state, sess):
    feed_dict = { self.value_net.states: [state] }
    preds = sess.run(self.value_net.predictions, feed_dict)
    return preds["logits"][0]

  def run_n_steps(self, n, sess):
    """
      Here the worker return transition tuple, local steps, and global steps every t_max (=5) actions.
      Then the global network gets update

    """
    transitions = []
    for _ in range(n):
      # Take a step
      action_probs, lstm_state = self._policy_net_predict(sess, self.state, *self.lstm_state)
      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
      next_state, reward, done, _ = self.env.step(action)
      self.lstm_state = lstm_state
      # self.env.render()
      # next_state = atari_helpers.atari_make_next_state(self.state, next_state)
      next_state = np.append(self.state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
      # Store transition
      transitions.append(Transition(
        state=self.state, action=action, reward=reward, next_state=next_state, done=done))

      # Increase local and global counters
      local_t = next(self.local_counter)
      global_t = next(self.global_counter)
      if global_t%20000==0:
        self.display_flag = True


      self.episode_local_step += 1

      if local_t % 500 == 0:
        tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))
        # print("Worker {} local step: {}, running {} of {} steps".format(
        #   self.name, self.episode_local_step, local_t, global_t))

      if done:
        # reset init state
        self.state = self.env.reset()
        # reset LSTMs memory
        lstm_c_init = np.zeros((1, 256), np.float32)
        lstm_h_init = np.zeros((1, 256), np.float32)
        self.lstm_state = [lstm_c_init, lstm_h_init]

        self.state = np.stack([self.state] * 4, axis=2)
        self.episode += 1
        self.episode_local_step = 0

        break
      else:
        if self.episode_local_step > 1000:
          # reset init state
          self.state = self.env.reset()
          # reset LSTMs memory
          lstm_c_init = np.zeros((1, 256), np.float32)
          lstm_h_init = np.zeros((1, 256), np.float32)
          self.lstm_state = [lstm_c_init, lstm_h_init]

          self.state = np.stack([self.state] * 4, axis=2)
          self.episode += 1
          self.episode_local_step = 0
        else:
          self.state = next_state



    return transitions, local_t, global_t

  def update(self, transitions, sess):
    """
    Updates global policy and value networks based on collected experience

    Args:
      transitions: A list of experience transitions
      sess: A Tensorflow session
    """

    # If we episode was not done we bootstrap the value from the last state
    reward = 0.0
    if not transitions[-1].done:
      reward = self._value_net_predict(transitions[-1].next_state, sess)

    # Accumulate minibatch exmaples
    states = []
    policy_targets = []
    value_targets = []
    actions = []

    for transition in transitions[::-1]:
      # The [::-1] slice reverses the list in the for loop
      # (but won't actually modify your list "permanently").

      reward = transition.reward + self.discount_factor * reward

      #>>> Policy target or target diff?
      policy_target = (reward - self._value_net_predict(transition.state, sess))
      # Accumulate updates
      states.append(transition.state)
      actions.append(transition.action)
      policy_targets.append(policy_target)
      value_targets.append(reward)

    # batch size = 5
    feed_dict = {
      self.policy_net.states: np.array(states),
      self.policy_net.targets: policy_targets,
      self.policy_net.actions: actions,
      self.value_net.states: np.array(states),
      self.value_net.targets: value_targets,
    }

    # Train the global estimators using local gradients
    global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run([
      self.global_step,
      self.policy_net.loss,
      self.value_net.loss,
      self.pnet_train_op,
      self.vnet_train_op,
      self.policy_net.summaries,
      self.value_net.summaries
    ], feed_dict)

    # Write summaries
    if self.summary_writer is not None:
      self.summary_writer.add_summary(pnet_summaries, global_step)
      self.summary_writer.add_summary(vnet_summaries, global_step)
      self.summary_writer.flush()

    return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries
