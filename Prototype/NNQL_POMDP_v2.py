from Env.BinaryEnvTest import MazeEnv
from collections import namedtuple

import random, time, numpy as np, sys, os, tensorflow as tf, itertools
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from matplotlib import pyplot as plt

valid_actions = [0, 1, 2, 3]


env = MazeEnv()

plt.ion()

class Q_Network():

    def __init__(self, scope, summary_dir0 = None):
        self.scope = scope
        self.keep_prob = 0.5
        self.fc1_num_outputs = 500
        self.fc2_num_outputs = 500
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
        self.X_tr = tf.placeholder(shape = [None, 12, 12, 4], dtype = tf.uint8, name = 'X')
        # Placeholder for approxiamate target (labels)
        self.y_tr = tf.placeholder(shape = [None], dtype = tf.float32, name = 'y')
        # Placeholder for action choices
        self.action_tr = tf.placeholder(shape = [None], dtype = tf.int32, name = 'action')

        # Input data normalization
        # N.A. for binary image

        # Flatten the input images and build fully connected layers
        X = tf.contrib.layers.flatten(tf.to_float(self.X_tr)/255)


        fc1 = tf.layers.dense(X, self.fc1_num_outputs, activation= tf.nn.relu, \
                              kernel_initializer = tf.random_normal_initializer(0.,0.3), \
                              bias_initializer= tf.constant_initializer(0.1))
       # fc1_dropout = tf.contrib.layers.dropout(fc1, self.keep_prob)

        fc2 = tf.layers.dense(fc1, self.fc2_num_outputs, activation=tf.nn.relu, \
                              kernel_initializer=tf.random_normal_initializer(0., 0.3), \
                              bias_initializer=tf.constant_initializer(0.1))

        fc2_dropout = tf.contrib.layers.dropout(fc2, self.keep_prob)

        self.q_val = tf.layers.dense(fc2_dropout, self.n_actions, \
                              kernel_initializer=tf.random_normal_initializer(0., 0.3), \
                              bias_initializer=tf.constant_initializer(0.1))

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
    #policy[np.argmax(q_val)] += 1 - epsilon
    policy = np.argmax(q_val)
    return policy



def Copy_Network(sess, network1, network2):
        # t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= network1.scope)
        # e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope= network2.scope)
        # copy_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(network1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(network2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        copy_op = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            copy_op.append(op)

        return sess.run(copy_op)


def Q_learning(sess,env, q_eval_net, target_net, num_episodes, replay_memory_size, replay_memory_initial_size, \
               target_net_update_interval, discounted_factor, epsilon_s, epsilon_f, batch_size, max_iter_num):

    # Initialize a MDP tuple
    MDP_tuple = namedtuple('MDP_tuple',['state', 'action', 'reward', 'next_state', 'terminate'])

    # Initialize replay memory
    replay_memory = []

    total_step = sess.run(tf.contrib.framework.get_global_step())

    total_step_ = 1

    epsilon_array = np.linspace(epsilon_s, epsilon_f, num_episodes)

    # Populate the replay memory with random states
    state = env.reset()
    state = np.stack([state] * 4, axis=2)

    for i in range(replay_memory_initial_size):

        policy = Policy_Fcn(sess, q_eval_net, state, env.n_actions, \
                               epsilon_array[min(total_step, num_episodes-1)])
        # action = np.random.choice(np.arange(env.n_actions), p = policy)
        action = np.random.choice(np.arange(env.n_actions))

        next_state, reward, done, _ = env.step(action)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

        replay_memory.append([state, action, reward, next_state, done])
        if done:
            state = env.reset()
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state
        if(i % 100 == 0):
            print("\r Populating replay memory {}% completed".format(
                100*float(i)/replay_memory_initial_size)),
        sys.stdout.flush()

    for i_episode in range(num_episodes):

        state= env.reset()
        state = np.stack([state] * 4, axis=2)
        loss = None
        transition = []
        # Maybe update the target estimator
        if i_episode % target_net_update_interval == 0:
            Copy_Network(sess, q_eval_net, target_net)
            print("\nCopied model parameters to target network.")




        for t in itertools.count():



            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({} of {}) Scene {} @ Episode {}/{}, loss: {}".format(
                t % max_iter_num, total_step_, total_step, t/max_iter_num+1, i_episode + 1, num_episodes, loss) ),
            sys.stdout.flush()

            policy = Policy_Fcn(sess, q_eval_net, state, env.n_actions,
                                epsilon_array[min(i_episode, num_episodes - 1)])
            if np.random.uniform() > epsilon_array[min(i_episode, num_episodes - 1)]:
                action = policy #np.random.choice(np.arange(env.n_actions), p=policy)
            else:
                action = np.random.choice(np.arange(env.n_actions))
            next_state, reward, done, _ = env.step(action)

            next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

            transition.append([state, action, reward, next_state, done])
            # If replay memory is full, first-in-first-out
            # if len(replay_memory) == replay_memory_size:
            #     replay_memory.pop(0)
            # replay_memory.append([state, action, reward, next_state, done])


            # Minibatch Training

            # Sample the training set from replay memory
            training_set = random.sample(replay_memory, batch_size)

            state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*training_set))

            # Use the "Q evaluation network" to estimate q values of the next states (of the training set)
            q_val_batch = q_eval_net.model_predict(sess, next_state_batch)

            # q_val_batch_max_idx = np.argmax(q_val_batch,axis = 1)
            #
            # q_val_target_batch = target_net.model_predict(sess, next_state_batch)

            # target_batch = reward_batch + \
            #                discounted_factor * np.invert(done_batch).astype(float) * \
            #                (q_val_target_batch[np.arange(batch_size), q_val_batch_max_idx] )

            target_batch = reward_batch + \
                        discounted_factor * np.invert(done_batch).astype(float) * (np.amax(q_val_batch, axis=1))

            # target_batch = reward_batch + discounted_factor * (np.amax(q_val_batch, axis=1))

            state_batch = np.array(state_batch)

            loss = q_eval_net.update_model(sess, state_batch, action_batch, target_batch)

            if (done):
                print ('Step = {}'.format(t%max_iter_num) )

                for ti in range(len(transition)):
                    if len(replay_memory) == replay_memory_size:
                        replay_memory.pop(0)
                    replay_memory.append(transition[ti])
                    total_step_ += 1


                break

            state = next_state
            total_step += 1

            if t > max_iter_num and t % max_iter_num == 1:
                state = env.reset()
                state = np.stack([state] * 4, axis=2)
                transition = []
                loss = None



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
    Q_learning(sess, env, q_eval_net, target_net, num_episodes = 3000, replay_memory_size = 100000,\
               replay_memory_initial_size = 10000, target_net_update_interval = 10, discounted_factor = 0.99, \
               epsilon_s = 1.0, epsilon_f = 0.0, batch_size = 32, max_iter_num = 500)
