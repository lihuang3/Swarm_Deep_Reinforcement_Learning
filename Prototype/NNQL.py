from Env.BinaryEnvTest import MazeEnv

import time, numpy as np, sys, os, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


np.random.seed(10)

env = MazeEnv()


class Q_Network():

    def __init__(self, fc_num_outputs, n_actions,scope = "Q-Network", summary_dir0 = None):
        self.scope = scope
        self.fc_num_outputs = fc_num_outputs
        self.n_actions = n_actions

        with tf.variable_scope(scope):
            self._build_neural_network()
            if summary_dir0:
                summary_dir = os.path.join(summary_dir0, "summary_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_neural_network(self):

        # Placeholder for input binary images
        self.X_tr = tf.placeholder(shape = [None, 12, 12], dtype = tf.uint8, name = 'X')
        # Placeholder for approxiamate target (labels)
        self.y_tr = tf.placeholder(shape = [None], dtype = tf.float32, name = 'y')
        # Placeholder for action choices
        self.action_tr = tf.placeholder(shape = [None], dtype = tf.int32, name = 'action')

        # Input data normalization
        # N.A. for binary image

        # Flatten the input images and build fully connected layers

        X = tf.contrib.layers.flatten(tf.to_float(self.X_tr))
        fc = tf.contrib.layers.fully_connected(X, self.fc_num_outputs, activation_fn=tf.nn.relu,\
                                               weights_initializer = tf.random_normal_initializer(), \
                                               biases_initializer = tf.zeros_initializer())
        self.q_eval = tf.contrib.layers.fully_connected(fc,self.n_actions, activation_fn=tf.nn.relu,\
                                               weights_initializer = tf.random_normal_initializer(), \
                                               biases_initializer = tf.zeros_initializer())

        # Make prediction
        # tf.gather_nd(params, indices): map elements in params to the output with given the indices order

        self.q_pred = tf.gather_nd(self.q_eval, tf.stack([tf.range(tf.shape(self.action_tr)[0]),self.action_tr], axis = 1))

        # Loss function
        self.loss_vector = tf.squared_difference(self.y_tr, self.q_pred)
        self.loss = tf.reduce_mean(self.loss_vector, name = 'TD_error')
        # Optimizer and train operations
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss, global_step = tf.contrib.framework.get_global_step() )


        # # Tensorboard Summary
        # self.summary = tf.summary.merger([
        #     tf.summary.scalar('loss', self.loss),
        #     tf.summary.histogram('lost_hist', self.loss_vector),
        #     tf.summary.histogram('q_eval_hist', self.q_eval),
        #     tf.summary.scaler('q_pred', self.q_pred)
        # ])


def PolicyFcn(sess, Q_Network, state, n_actions, epsilon):
    # Initialize uniform policy
    policy = np.ones(n_actions, dtype = float)*epsilon/n_actions
    # Augment the state since training input is of shape [?, 12, 12]
    state = np.expand_dims(state,0)
    # Evaluate q values and squeeze the 1-element dimension since the output is of shape [?, 4]
    q_val = sess.run(Q_Network.q_eval, feed_dict = {Q_Network.X_tr: state})

    # Update the priority action probability
    policy[np.argmax(q_val)] += 1 - epsilon
    return policy, q_val

def TargetFcn(sess, Q_Network, state, target, action):
    state = np.expand_dims(state, 0)
    loss, _ = sess.run([Q_Network.loss, Q_Network.train_op], \
             feed_dict = {Q_Network.X_tr: state, Q_Network.y_tr: target, Q_Network.action_tr: action})
    return loss

def Q_learning(sess,env, q_eval, q_target, num_episodes, replay_memory_size, replay_memory_initial_size, \
               q_target_net_update, discounted_factor, epsilon_s, epsilon_f, epsilon_delay, batch_size):
    return


state = env.reset()
action = np.random.randint(4,size = 1)

q_eval_network = Q_Network(fc_num_outputs = 50, n_actions = 4)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        state, reward, done, _ = env.step(action)
        policy, q_val = PolicyFcn(sess, q_eval_network,state, n_actions=4,epsilon = 0.5)
        target = reward + 0.9*(max(q_val))
        action = np.random.choice(np.arange(4), 1, p = policy)
        TargetFcn(sess, q_eval_network, state, target, action)
        if done:
            state = env.reset()
            action = np.random.randint(4, size=1)
        if i%50 ==1 or done:
            print "\n Action = {}, Reward = {} \n Policy = {}".format(action, reward, q_val)


