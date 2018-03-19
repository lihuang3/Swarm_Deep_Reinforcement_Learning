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

  def __init__(self, feature_space, action_space):

    self.state = X1 = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X1")
    self.next_state = X2 = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X2")

    X1 = tf.to_float(X1)/255.0
    X2 = tf.to_float(X2)/255.0
    batch_size = tf.shape[X1][0]

    #  feature encoding phi1, phi2
    phi1 = build_shared_network(X1)
    phi2 = build_shared_network(X2)

    # Xt is the time series input for LSTMs--> augment a fake batch dimension of 1 to
    # do LSTMs over time dimension
    Xt = tf.expand_dims(phi1, [0])

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

    self.value_logits = tf.reshape(tf.layers.dense(inputs=psi, num_outputs=1, scope="value_fcn", activation=None),[-1])

    self.policy_logits = tf.layers.dense(inputs=psi, num_outputs=action_space, scope="policy_fcn", activation=None)

    self.log_probs = tf.nn.log_softmax(self.policy_logits)

    self.probs = tf.nn.softmax(logits=self.policy_logits, dim=-1)[0,:]

    self.actions = tf.one_hot(tf.squeeze(input=tf.multinomial(logits=self.probs), axis=1), action_space)[0,:]

    # We add entropy to the loss to encourage exploration
    self.entropy = -tf.reduce_mean(tf.reduce_sum(self.probs * tf.log(self.probs), 1), name="entropy")

    # Policy targets
    self.advantage = tf.placeholder(shape=[None], dtype=tf.float32)

    # Value fcn targets
    self.reward = tf.placeholder(shape=[None], dtype=tf.float32)

    policy_loss = -tf.reduce_mean(tf.reduce_sum(self.log_probs * self.actions, axis=1) * self.advantage)

    value_fcn_loss = 0.5 * tf.squared_difference(self.value_logits, self.reward)

    # Final A3C loss
    self.loss = policy_loss + 0.5 * value_fcn_loss + 0.01 * self.entropy

    self.optimizer = tf.train.RMSPropOptimizer(0.0025, 0.99, 0.0, 1e-6)

    self.grads_and_vars = self.optimizer.compute_gradients(self.loss)

    self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]


    # Inverse dynamics model g(phi1, phi2) --> pred_act
    g = tf.concat([phi1, phi2], axis=1)

    g = tf.layers.dense(inputs=g, num_outputs=256, activation=tf.nn.relu)

    inv_logits = tf.layers.dense(inputs=g, num_outputs=action_space, activation=None)

    action_index = tf.argmax(self.actions, axis=1)

    self.inv_loss = tf.reduce_mean(tf.softmax_cross_entropy_with_logits(inv_logits, self.actions), name="inv_loss")

    # Forward dynamics model f(phi1, action) --> pred_phi2
    f = tf.concat([phi1, self.actions], axis=1)

    f = tf.layers.dense(inputs=f, num_outputs=256, activation=tf.nn.relu)

    pred_phi2 = tf.layers.dense(inputs=f, num_outputs=tf.shape(phi1)[1], activation=None)

    self.fwd_loss = tf.reduce_mean(tf.squared_difference(phi2, pred_phi2))

    self.pred_loss = 0.8 * self.inv_loss + 0.2 * self.fwd_loss

    self.pred_grads_and_vars =self.optimizer.compute_gradients(self.pred_loss)

    # Traning op
    self.grads_and_vars = self.pred_grads_and_vars + self.grads_and_vars

    self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                                                   global_step=tf.train.get_global_step())


  def reset_lstm(self):
    return self.state_init

  def make_action(self, observation, lstm_cin, lstm_hin):
    sess = tf.get_default_session()
    return sess.run([self.actions, self.state_out],
                    feed_dict={self.state: observation, self.state_int[0]:lstm_cin, self.state_init[1]:lstm_hin})

  def make_train_op(self):
    sess = tf.get_default_session()

