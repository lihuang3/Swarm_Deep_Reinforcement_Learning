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
# from estimators import cnn_lstm
from estimators2 import cnn_lstm
from estimators2 import fwd_inv_model

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "value_logits",
                                                   "next_state", "done", "feature", "episode_end"])


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
def make_train(local_estimator, global_estimator):
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
  def __init__(self, name, start_time, saver, checkpoint_path, env, global_net, global_model,
               global_counter, global_success, discount_factor=0.99, summary_writer=None, max_global_steps=None):
    self.name = name
    self.start_time = start_time
    self.saver = saver
    self.checkpoint_path = checkpoint_path
    self.discount_factor = discount_factor
    self.max_global_steps = max_global_steps
    self.global_step = tf.train.get_global_step()
    self.global_net = global_net
    self.global_model = global_model
    self.global_counter = global_counter
    self.local_counter = itertools.count()
    self.global_success = global_success
    self.summary_writer = summary_writer
    self.env = env
    self.episode = 1
    self.episode_local_step = 0
    self.display_flag = False
    self.saver_flag = False
    # Create local policy/value nets that are not updated asynchronously
    with tf.variable_scope(name):
      self.local_net = cnn_lstm(feature_space=256, action_space=4)
      with tf.variable_scope("predictor"):
        self.local_model = fwd_inv_model(feature_space=256, action_space=4)

    # Op to copy params from global policy/valuenets
    self.copy_params_op = make_copy_params_op(
      tf.contrib.slim.get_variables(scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
      tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.GraphKeys.TRAINABLE_VARIABLES))

    self.train_global_net = make_train(self.local_net, self.global_net)
    self.train_global_model = make_train(self.local_model, self.global_model)

    self.state = None

    self.MDP = []

  def run(self, sess, coord, t_max):
    with sess.as_default(), sess.graph.as_default():
      # Initial state
      self.state = self.env.reset()
      self.state = np.expand_dims(self.state, axis=2)
      # self.state = np.stack([self.state] * 4, axis=2)
      # Init LSTMs state
      self.lstm_state = self.local_net.reset_lstm()

      try:
        while not coord.should_stop():
          # Copy Parameters from the global networks
          sess.run(self.copy_params_op)

          # Collect some experience
          transitions, local_t, global_t = self.run_n_steps(t_max, sess)

          self.MDP.extend(transitions)

          if self.max_global_steps is not None and global_t >= self.max_global_steps:
            tf.logging.info("Reached global step {}. Stopping.".format(global_t))
            coord.request_stop()
            return

          # Update the global networks
          loss, entropy, pi_loss, value_loss, _, fwd_loss, inv_loss, _=self.update(transitions, sess)

          if transitions[-1].episode_end:
            # if transitions[-1].done:
            #   for _ in range(5):
            #     loss, entropy, pi_loss, value_loss, _, fwd_loss, inv_loss, _ = self.update(self.MDP, sess)
            self.MDP = []

          if self.display_flag:
            training_time = int(time.time()-self.start_time)
            print("{} d {} hr {} min, step = {}/{}, {}, Ep {}, a3c_loss = {:.4E}, "
                  "entropy = {:.4E}, piloss = {:.4E}, vloss = {:.4E}, fwd = {:.4E}, inv = {:.4E}".format(training_time/86400,
                                    (training_time/3600)%24, (training_time/60)%60, local_t, global_t, self.name, self.episode,
                                                                  loss, entropy, pi_loss, value_loss, fwd_loss, inv_loss))
            self.display_flag = False

          if self.saver_flag:
            self.saver.save(sess, self.checkpoint_path)
            self.saver_flag = True



      except tf.errors.CancelledError:
        return

  def run_n_steps(self, n, sess):
    """
      Here the worker return transition tuple, local steps, and global steps every t_max (=5) actions.
      Then the global network gets update

    """
    transitions = []
    for _ in range(n):
      # Take a step
      fetched = self.local_net.make_action(self.state, self.lstm_state[0], self.lstm_state[1])
      action_onehot, policy_probs, value_logits, self.lstm_state = fetched[0], fetched[1], fetched[2], fetched[3:]
      action = action_onehot.argmax()
      next_state, reward, done, _ = self.env.step(action)
      # self.env.render()
      next_state = np.expand_dims(next_state, 2)
      extrinsic_reward = reward
      intrinsic_reward = self.local_model.intrinsic_reward(self.state, next_state, action_onehot)
      reward += intrinsic_reward

      # next_state = atari_helpers.atari_make_next_state(self.state, next_state)
      # next_state = np.append(self.state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
      # Store transition
      self.episode_local_step += 1
      if self.episode_local_step==600 or done:
        episode_end = True
      else:
        episode_end = False

      transitions.append(Transition(
        state=self.state, action=action_onehot, reward=reward, value_logits=value_logits[0], next_state=next_state,
        done=done, feature=self.lstm_state, episode_end=episode_end))

      # Increase local and global counters
      local_t = next(self.local_counter)
      global_t = next(self.global_counter)

      if global_t%5000==0:
        self.display_flag=True
      if global_t%200000==0:
        self.saver_flag=True

      if self.name=="worker_0" and self.episode_local_step%200 ==0:
        print ("Step {} of Ep {}, policy probs = {}, value logits = {:.4E}, inrwd = {:.4E}, exrwd = {}".format(
          self.episode_local_step, self.episode, policy_probs, value_logits[0], 100*intrinsic_reward, extrinsic_reward))

      if local_t % 500 == 0:
        tf.logging.info("{}: local Step {}, global step {}".format(self.name, local_t, global_t))
        # print("Worker {} local step: {}, running {} of {} steps".format(
        #   self.name, self.episode_local_step, local_t, global_t))

      if done or self.episode_local_step==600:
        if done:
          global_success = next(self.global_success)
          print("Success times: {}, {}, Episode step = {}, Episode = {},".format(global_success, self.name, self.episode_local_step, self.episode))
          print("    ")
        # reset init state
        self.state = self.env.reset()
        self.state = np.expand_dims(self.state, axis=2)
        # reset LSTMs memory
        self.lstm_state = self.local_net.reset_lstm()
        # self.state = np.stack([self.state] * 4, axis=2)
        self.episode += 1
        self.episode_local_step = 0

        break
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
      reward = self.local_net.pred_value(observation=transitions[-1].next_state,
                                         lstm_cin=transitions[-1].feature[0], lstm_hin=transitions[-1].feature[1])
      reward = reward[0]
    # Accumulate minibatch exmaples
    states = []
    next_states = []
    policy_targets = []
    value_targets = []
    actions = []
    features = []

    for transition in transitions[::-1]:
      # The [::-1] slice reverses the list in the for loop
      # (but won't actually modify your list "permanently").

      # Value target
      reward = transition.reward + self.discount_factor * reward
      # Policy target (advantage)
      policy_target = (reward - transition.value_logits)
      # Accumulate updates
      states.append(transition.state)
      next_states.append(transition.next_state)
      actions.append(transition.action)
      policy_targets.append(policy_target)
      value_targets.append(reward)

    #Todo: try the last feature as lstm state in [c_in, h_in]
    features = transitions[-1].feature



    # batch size = 5
    feed_dict = {
      self.local_net.state: np.array(states),
      self.local_net.acs: actions,
      self.local_net.state_in[0]: np.array(features[0]),
      self.local_net.state_in[1]: np.array(features[1]),
      self.local_net.advantage: policy_targets,
      self.local_net.reward: value_targets,
      self.local_model.state: np.array(states),
      self.local_model.acs: actions,
      self.local_model.next_state: np.array(next_states),

    }
    # Train the global estimators using local gradients
    # Use dummy nodes to skip unnecessary communication if the nodes are only needed for dependencies but not output

    global_step, loss, entropy, pi_loss, value_loss, _, summaries1,\
      fwd_loss, inv_loss, _, summaries2 = sess.run(
      [
        self.global_step,
        self.local_net.loss,
        self.local_net.entropy,
        self.local_net.policy_loss,
        self.local_net.value_fcn_loss,
        self.train_global_net,
        self.local_net.summaries,
        self.local_model.fwd_loss,
        self.local_model.inv_loss,
        self.train_global_model,
        self.local_model.summaries
      ], feed_dict)


    # Write summaries
    if self.summary_writer is not None:
      self.summary_writer.add_summary(summaries1, global_step)
      self.summary_writer.add_summary(summaries2, global_step)
      self.summary_writer.flush()

    return loss, entropy, pi_loss, value_loss, summaries1, fwd_loss, inv_loss, summaries2
