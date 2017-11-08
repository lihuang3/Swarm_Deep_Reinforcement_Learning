from Env.BinaryEnvTest import MazeEnv
from collections import namedtuple

import random, time, numpy as np, sys, os, tensorflow as tf, itertools
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

valid_actions = [0, 1, 2, 3]

np.random.seed(10)

env = MazeEnv()


class Q_Network():

    def __init__(self, scope, summary_dir0 = None):
        self.scope = scope
        self.fc1_num_outputs = 1000
        self.fc2_num_outputs = 100
        self.n_actions = env.n_actions

        with tf.variable_scope(scope):
            self._build_model()
            if summary_dir0:
                summary_dir = os.path.join(summary_dir0, "summary_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):

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
        fc1 = tf.contrib.layers.fully_connected(X, self.fc1_num_outputs, activation_fn=tf.nn.relu,\
                                               weights_initializer = tf.random_normal_initializer(), \
                                               biases_initializer = tf.zeros_initializer())

        fc2 = tf.contrib.layers.fully_connected(fc1, self.fc2_num_outputs, activation_fn=tf.nn.relu,\
                                               weights_initializer = tf.random_normal_initializer(), \
                                               biases_initializer = tf.zeros_initializer())

        self.q_val = tf.contrib.layers.fully_connected(fc2, self.n_actions, activation_fn=tf.nn.relu,\
                                               weights_initializer = tf.random_normal_initializer(), \
                                               biases_initializer = tf.zeros_initializer())

        # Make prediction
        # tf.gather_nd(params, indices): map elements in params to the output with given the indices order
        action_stack = tf.stack([tf.range(tf.shape(self.action_tr)[0]),self.action_tr], axis = 1)
        self.q_pred = tf.gather_nd(self.q_val, action_stack)

        # Loss function
        self.loss_vector = tf.squared_difference(self.y_tr, self.q_pred)
        self.loss = tf.reduce_mean(self.loss_vector, name = 'TD_error')
        # Optimizer and train operations
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step = tf.contrib.framework.get_global_step() )

    # Predict q values
    def model_predict(self, sess, state):
        if state.ndim < 3:
            state = np.expand_dims(state, 0)
        return sess.run(self.q_val, feed_dict = {self.X_tr: state})

    # Update model parameters
    def update_model(self, sess, state, action, target):

        loss, _ = sess.run([self.loss, self.train_op], \
                           feed_dict={self.X_tr: state, self.action_tr: action, self.y_tr: target})
        return loss
        # # Tensorboard Summary
        # self.summary = tf.summary.merger([
        #     tf.summary.scalar('loss', self.loss),
        #     tf.summary.histogram('lost_hist', self.loss_vector),
        #     tf.summary.histogram('q_eval_hist', self.q_eval),
        #     tf.summary.scaler('q_pred', self.q_pred)
        # ])

# Epsilon-greedy action selection policy
def Policy_Fcn(sess, network, state, n_actions, epsilon):
    # Initialize uniform policy
    policy = np.ones(n_actions, dtype = float)*epsilon/n_actions
    # Augment the state since training input is of shape [?, 12, 12]
    state = np.expand_dims(state,0)
    # Evaluate q values and squeeze the 1-element dimension since the output is of shape [?, 4]
    q_val = network.model_predict(sess, state)[0]

    # Update the priority action probability
    policy[np.argmax(q_val)] += 1 - epsilon
    return policy



def Copy_Network(sess, network1, network2):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= network1.scope)
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= network2.scope)
        copy_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        return sess.run(copy_op)


def Q_learning(sess,env, q_eval_net, target_net, num_episodes, replay_memory_size, replay_memory_initial_size, \
               target_net_update_interval, discounted_factor, epsilon_s, epsilon_f, batch_size):

    # Initialize a MDP tuple
    MDP_tuple = namedtuple('MDP_tuple',['state', 'action', 'reward', 'next_state', 'terminate'])

    # Initialize replay memory
    replay_memory = []

    total_step = sess.run(tf.contrib.framework.get_global_step())


    epsilon_array = np.linspace(epsilon_s, epsilon_f, num_episodes)

    # Populate the replay memory with random states
    state = env.reset()
    for i in range(replay_memory_initial_size):

        policy = Policy_Fcn(sess, q_eval_net, state, env.n_actions, \
                               epsilon_array[min(total_step, num_episodes-1)])
        action = np.random.choice(np.arange(env.n_actions), p = policy)
        next_state, reward, done, _ = env.step(action)
        replay_memory.append([state, action, reward, next_state, done])
        if done:
            state = env.reset()
        else:
            state = next_state
        if(i % 100 ==0):
            print("\r Populating replay memory {}% completed".format(
                100*float(i)/replay_memory_initial_size)),
        sys.stdout.flush()

    for i_episode in range(num_episodes):

        state = env.reset()
        loss = None

        for t in itertools.count():

            # Maybe update the target estimator
            if total_step % target_net_update_interval == 0:
                Copy_Network(sess, q_eval_net, target_net)
                print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                t, total_step, i_episode + 1, num_episodes, loss) ),
            sys.stdout.flush()

            policy = Policy_Fcn(sess, q_eval_net, state, env.n_actions,
                                epsilon_array[min(total_step, num_episodes - 1)])
            action = np.random.choice(np.arange(env.n_actions), p=policy)
            next_state, reward, done, _ = env.step(action)

            # If replay memory is full, first-in-first-out
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append([state, action, reward, next_state, done])

            if done:
                state = env.reset()
            else:
                state = next_state

            # Minibatch Training

            # Sample the training set from replay memory
            training_set = random.sample(replay_memory, batch_size)

            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*training_set))

            # Use the "Q evaluation network" to estimate q values of the next states (of the training set)
            q_val_batch = q_eval_net.model_predict(sess, next_state_batch)

            target_batch = reward_batch + \
                           discounted_factor*np.invert(done_batch).astype(float)*( np.amax(q_val_batch, axis = 1,))

            loss = q_eval_net.update_model(sess, state_batch, action_batch, target_batch)

            if (done) or (t > 200):
                print ('  Step = {} Done = {}'.format(t, done))
                break

            state = next_state
            total_step += 1


#=======================
# Main Program
#=======================

tf.reset_default_graph()

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Environment Initialization
state = env.reset()

action = np.random.randint(4,size = 1)

q_eval_net= Q_Network(scope = 'q_eval_net')

target_net = Q_Network(scope = 'target_net')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Q_learning(sess, env, q_eval_net, target_net, num_episodes = 10000, replay_memory_size = 100000,\
               replay_memory_initial_size = 10000, target_net_update_interval = 4000, discounted_factor = 0.99, \
               epsilon_s = 1.0, epsilon_f = 0.1, batch_size = 32)
