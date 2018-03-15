import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

def build_shared_network(X):
  """
  Builds a 3-layer network conv -> conv -> fc as described
  in the A3C paper. This network is shared by both the policy and value net.

  Args:
    X: Inputs
    add_summaries: If true, add layer summaries to Tensorboard.

  Returns:
    Final layer activations.
  """

  # Three convolutional layers
  # conv1 = tf.contrib.layers.conv2d(
  #   X, 16, 8, 4, activation_fn=tf.nn.relu, scope="conv1")
  # conv2 = tf.contrib.layers.conv2d(
  #   conv1, 32, 4, 2, activation_fn=tf.nn.relu, scope="conv2")

  # # Fully connected layer
  # fc1 = tf.contrib.layers.fully_connected(
  #   inputs=tf.contrib.layers.flatten(conv2),
  #   num_outputs=256,
  #   scope="fc1")

  # Three convolutional layers
  conv1 = tf.layers.conv2d(
    input=X, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu, scope="conv1")
  conv2 = tf.layers.conv2d(
    input=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, scope="conv2")
  conv3 = tf.layers.conv2d(
    input=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, scope="conv3")

  # Fully connected layer
  fc1 = tf.layers.dense(
    inputs=tf.layers.flatten(conv3), num_outputs=512, scope="fc1", activation=tf.nn.relu)

  # if add_summaries:
  #   tf.contrib.layers.summarize_activation(conv1)
  #   tf.contrib.layers.summarize_activation(conv2)
  #   tf.contrib.layers.summarize_activation(fc1)

  return fc1

class cnn_lstm():
  """
  Builds a lstm rnn cnn_fc -> lstm -> feature space
  and output policy logits and value fcn loss

  Args:
  """

  def __init__(self, scope, feature_space, action_space):
    self.state = X = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")

    X = tf.to_float(X)/255.0
    batch_size = tf.shape[X][0]
    with tf.variable_scope(scope, reuse=True):
       phi = build_shared_network(X)

    # Xt is the time series input for LSTMs--> augment a fake batch dimension of 1 to
    # do LSTMs over time dimension
    Xt = tf.expand_dims(phi, [0])

    # Initialize RNN-LSTMs cell with feature space size = 256
    lstm = rnn.BasicLSTMCell(num_units=feature_space, forget_bias=1.0, state_is_tuple=True)
    # Todo: is this time sequence step size?
    step_size = tf.shape(Xt)[:1]

    # Reset lstm memeory cells
    # lstm cell is a tuple = [c_in, h_in]
    c_init = np.zeros((1, lstm.state_size.c), np.float32)
    h_init = np.zeros((1, lstm.state_size.h), np.float32)
    self.state_init = [c_init, h_init]

    # Pass lstm state from last training to the current training
    c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
    h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
    self.state_in = [c_in, h_in]

    init_tuple = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)

    lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm, Xt, initial_state=init_tuple, sequence_length=batch_size, time_major=False)

    lstm_c, lstm_h = lstm_state

    # RNN feature-space state
    psi = tf.reshape(lstm_outputs, [-1, feature_space])
    self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

    self.value_fcn = tf.layers.dense(inputs=psi, num_outputs=1, scope="value_fcn", activation=None)

    self.policy_logits = tf.layers.dense(inputs=psi, num_outputs=action_space, scope="policy_fcn", activation=None)
    self.policy = tf.nn.softmax(logits=self.policy_logits, dim=-1)[0,:]
    self.action = tf.one_hot(tf.squeeze(input=tf.multinomial(logits=self.policy), axis=1), action_space)[0,:]


  def reset_lstm(self):
    return self.state_init

  def predictions(self, env_state, lstm_cin, lstm_hin):
    sess = tf.get_default_session()
    return sess.run([self.value_fcn, self.action],
                    feed_dict={self.state: env_state, self.state_int[0]:lstm_cin, self.state_init[1]:lstm_hin})




class PolicyEstimator():
  """
  Policy Function approximator. Given a observation, returns probabilities
  over all possible actions.

  Args:
    num_outputs: Size of the action space.
    reuse: If true, an existing shared network will be re-used.
    trainable: If true we add train ops to the network.
      Actor threads that don't update their local models and don't need
      train ops would set this to false.
  """

  def __init__(self, num_outputs, reuse=False, trainable=True):
    self.num_outputs = num_outputs

    # Placeholders for our input
    # Our input are 4 RGB frames of shape 160, 160 each
    self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
    # The TD target value
    self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
    # Integer id of which action was selected
    self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

    # LSTM feature states:
    self.c_in = tf.placeholder(tf.float32, [1, 256], name='c_in')
    self.h_in = tf.placeholder(tf.float32, [1, 256], name='h_in')

    # Normalize
    X = tf.to_float(self.states) / 255.0
    batch_size = tf.shape(self.states)[0]

    # Graph shared with Value Net
    with tf.variable_scope("shared", reuse=reuse):
      # Feature space layer output
      lstm_output, lstm_state = build_shared_network(X, self.c_in, self.h_in, add_summaries=(not reuse))

    with tf.variable_scope("policy_net"):
      self.logits = tf.contrib.layers.fully_connected(lstm_output, num_outputs, activation_fn=None)
      self.probs = tf.nn.softmax(self.logits) + 1e-8

      self.predictions = {
        "logits": self.logits,
        "probs": self.probs,
        "features": lstm_state
      }

      # We add entropy to the loss to encourage exploration
      self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs), 1, name="entropy")
      self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")

      # Get the predictions for the chosen actions only
      gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
      self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)

      #>>> Is this cross entropy?
      self.losses = - (tf.log(self.picked_action_probs) * self.targets + 0.01 * self.entropy)
      self.loss = tf.reduce_sum(self.losses, name="loss")

      tf.summary.scalar(self.loss.op.name, self.loss)
      tf.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
      tf.summary.histogram(self.entropy.op.name, self.entropy)

      if trainable:
        #>>> Todo: why do you extract the gradients but not train it directly using minimize?
        # self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        # Get gradients and variables from the optimizer
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
          global_step=tf.train.get_global_step())

    # Merge summaries from this network and the shared network (but not the value net)
    var_scope_name = tf.get_variable_scope().name
    summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
    #>>> Todo: why is'summaries' defined twice?
    sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
    sumaries = [s for s in summary_ops if var_scope_name in s.name]
    self.summaries = tf.summary.merge(sumaries)


class ValueEstimator():
  """
  Value Function approximator. Returns a value estimator for a batch of observations.

  Args:
    reuse: If true, an existing shared network will be re-used.
    trainable: If true we add train ops to the network.
      Actor threads that don't update their local models and don't need
      train ops would set this to false.
  """

  def __init__(self, reuse=False, trainable=True):
    # Placeholders for our input
    # Our input are 4 RGB frames of shape 160, 160 each
    self.states = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
    # The TD target value
    self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

    X = tf.to_float(self.states) / 255.0

    # Graph shared with Value Net
    with tf.variable_scope("shared", reuse=reuse):
      fc1 = build_shared_network(X, add_summaries=(not reuse))

    with tf.variable_scope("value_net"):
      self.logits = tf.contrib.layers.fully_connected(
        inputs=fc1,
        num_outputs=1,
        activation_fn=None)
      self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")

      self.losses = tf.squared_difference(self.logits, self.targets)
      self.loss = tf.reduce_sum(self.losses, name="loss")

      self.predictions = {
        "logits": self.logits
      }

      # Summaries
      prefix = tf.get_variable_scope().name
      tf.summary.scalar(self.loss.name, self.loss)
      tf.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
      tf.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
      tf.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
      tf.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
      tf.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
      tf.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
      tf.summary.histogram("{}/reward_targets".format(prefix), self.targets)
      tf.summary.histogram("{}/values".format(prefix), self.logits)

      if trainable:
        # self.optimizer = tf.train.AdamOptimizer(1e-4)
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
        self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
          global_step=tf.train.get_global_step())

    var_scope_name = tf.get_variable_scope().name
    summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
    sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
    sumaries = [s for s in summary_ops if var_scope_name in s.name]
    self.summaries = tf.summary.merge(sumaries)
