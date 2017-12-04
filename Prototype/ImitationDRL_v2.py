from Env.RGBEnv_v1 import MazeEnv
from collections import namedtuple

import random, time, numpy as np, sys, os, tensorflow as tf, itertools
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from decimal import Decimal

'''
This imitation learning uses grayscale images as the input. We first populate 
the memory with MDP tuples of the expert (benchmark). Then, we train the CNN for 10 episodes to 
intialize. Then switch to DRL to improve the benchmark algorithm. Note this test does not intend for
general robots initial distributions. We use the same initial distributin each episode. 
'''


valid_actions = [0, 1, 2, 3]


env = MazeEnv()


class Q_Network():

    def __init__(self, scope, summary_dir0 = None):
        self.scope = scope
        self.keep_prob = 0.5
        self.fc_num_outputs = 1024
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
        self.X_tr = tf.placeholder(shape = [None, 84, 84, 4], dtype = tf.uint8, name = 'X')
        # Placeholder for approxiamate target (labels)
        self.y_tr = tf.placeholder(shape = [None], dtype = tf.float32, name = 'y')
        # Placeholder for action choices
        self.action_tr = tf.placeholder(shape = [None], dtype = tf.int32, name = 'action')
        # Placeholder for expert actions (one-hot)
        self.expert_action = tf.placeholder(shape = [None, 4], dtype = tf.float32, name = 'expert_action')

        # Input data normalization
        X = tf.to_float(self.X_tr)/255

        # Convolutional layer #1
        conv1 = tf.layers.conv2d(
            inputs=X,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu,
            strides=2)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.relu)

        # Pooling layer #2
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)



        # Flatten the input images and build fully connected layers
        flattened_layer = tf.contrib.layers.flatten(pool2)


        fc = tf.layers.dense(flattened_layer, self.fc_num_outputs, activation= tf.nn.relu, \
                              kernel_initializer = tf.random_normal_initializer(0.,0.3), \
                              bias_initializer= tf.constant_initializer(0.1))
        fc_dropout = tf.contrib.layers.dropout(fc, self.keep_prob)


        self.q_val = tf.layers.dense(fc_dropout, self.n_actions,  \
                              kernel_initializer=None, \
                              bias_initializer=tf.zeros_initializer())


        # Make prediction
        # tf.gather_nd(params, indices): map elements in params to the output with given the indices order
        action_stack = tf.stack([tf.range(tf.shape(self.action_tr)[0]),self.action_tr], axis = 1)
        self.q_pred = tf.gather_nd(self.q_val, action_stack)

        # Loss function
        self.loss_vector = tf.squared_difference(self.y_tr, self.q_pred)
        self.loss = tf.reduce_mean(self.loss_vector, name = 'TD_error')


        # Loss function (for expert demo)
        self.losses_exp = tf.squared_difference(self.expert_action, self.q_val)
        self.loss_exp = tf.reduce_sum(self.losses_exp, name = 'Supervised_error')
        #self.loss_exp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.expert_action, logits=self.q_val))

        # Optimizer and train operations
        # self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.optimizer = tf.train.AdamOptimizer(0.0001)
        self.train_op = self.optimizer.minimize(self.loss, global_step = tf.train.get_global_step() )
        self.train_op_expert_demo = self.optimizer.minimize(self.loss_exp, global_step = tf.train.get_global_step() )

        # Summaries for Tensorboard
        self.global_step = tf.train.get_global_step()

        # tf.summary.scalar("loss", self.loss_exp),
        # tf.summary.histogram("loss_hist", self.losses_exp),
        tf.summary.histogram("q_values_hist", self.q_pred),
        tf.summary.scalar("max_q_value", tf.reduce_max(self.q_pred))

        self.summaries = tf.summary.merge_all()

    # Predict q values
    def model_predict(self, sess, state):
        if state.ndim < 3:
            state = np.expand_dims(state, 0)
        return sess.run(self.q_val, feed_dict = {self.X_tr: state})


    # Update model parameters
    def update_model(self, sess, state, action, target):
        if target.ndim==2:
            global_step, summaries, loss, _ = sess.run([self.global_step, self.summaries, self.loss_exp, self.train_op_expert_demo], \
                               feed_dict={self.X_tr: state, self.action_tr: action, self.expert_action: target})
        else:
            global_step, summaries, loss, _ = sess.run([self.global_step, self.summaries, self.loss, self.train_op], \
                           feed_dict={self.X_tr: state, self.action_tr: action, self.y_tr: target})


        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


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
               target_net_update_interval, discounted_factor, epsilon_s, epsilon_f, batch_size, max_iter_num, expert_demo_num_episodes):

    # Create directories for the experiments and checkpoints
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, 'model')

    # Create a saver object
    saver = tf.train.Saver()
    # Load the latest checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print('Loading the latest checkpoint from ... \n {}'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Initialize a MDP tuple
    # MDP_tuple = namedtuple('MDP_tuple',['state', 'action', 'reward', 'next_state', 'terminate'])

    # Initialize replay memory
    replay_memory = []

    total_step = sess.run(tf.train.get_global_step())

    total_step_ = 1

    epsilon_array = np.linspace(epsilon_s, epsilon_f, num_episodes)

    # Populate the replay memory with random states
    state = env.reset()
    # Stack 4 successive frames for POMDP
    state = np.stack([state] * 4, axis=2)

    # Initialize the target robot loction for the expert demo
    robot_loc = []

    for i in range(replay_memory_initial_size):

        # policy = Policy_Fcn(sess, q_eval_net, state, env.n_actions, \
        #                        epsilon_array[min(total_step, num_episodes-1)])
        # # action = np.random.choice(np.arange(env.n_actions), p = policy)
        # action = np.random.choice(np.arange(env.n_actions))
        #
        # next_state, reward, done, _ = env.step(action)

        action, robot_loc = env.expert(robot_loc)

        next_state, reward, done, _ = env.step(action)

        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)

        replay_memory.append([state, action, reward, next_state, done])

        if done:
            state = env.reset()
            robot_loc = []

            # Stack 4 successive frames for POMDP
            state = np.stack([state] * 4, axis=2)
        else:
            state = next_state
        if(i % 100 == 0):
            print("\rPopulating replay memory {}% completed".format(
                100*float(i)/replay_memory_initial_size)),
        sys.stdout.flush()

    print("\r Populating replay memory 100% completed")

    for i_episode in range(num_episodes):
        # Save the current checkpoint
        saver.save(sess, checkpoint_path)
        episode_summary = tf.Summary()
        state= env.reset()
        # Stack 4 successive frames for POMDP
        state = np.stack([state] * 4, axis=2)
        loss = None
        transition = []

        # Initialize the target robot loction for the expert demo
        robot_loc = []

        loss = np.NaN
        local_loss = None
        # Maybe update the target estimator
        if (i_episode+1) % target_net_update_interval == 0:
            Copy_Network(sess, q_eval_net, target_net)
            print("\nCopied model parameters to target network.")


        for t in itertools.count():

            # Print out which step we're on, useful for debugging.
            print("\rStep {} ({} of {}) Scene {} @ Episode {}/{}, loss: {}".format(
                t % max_iter_num, total_step_, total_step, t/max_iter_num+1, i_episode + 1, \
                num_episodes, local_loss) ),
            sys.stdout.flush()


            if (i_episode+1)<=expert_demo_num_episodes:
                action, robot_loc = env.expert(robot_loc)
                if (i_episode+1)%5 ==0 and i_episode>1:
                    policy = Policy_Fcn(sess, q_eval_net, state, env.n_actions,
                                        epsilon_array[min(i_episode, num_episodes - 1)])
                    action = policy
            else:

                policy = Policy_Fcn(sess, q_eval_net, state, env.n_actions,
                                        epsilon_array[min(i_episode, num_episodes - 1)])
                #action = policy
                if np.random.uniform() > epsilon_array[min(i_episode, num_episodes - 1)]:
                    action = policy #np.random.choice(np.arange(env.n_actions), p=policy)
                else:
                    action = np.random.choice(np.arange(env.n_actions))


            next_state, reward, done, _ = env.step(action)

            # Stack 4 successive frames for POMDP
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

            q_val_batch = target_net.model_predict(sess, next_state_batch)
            target_batch = reward_batch + \
                           discounted_factor * np.invert(done_batch).astype(float) * (np.amax(q_val_batch, axis=1))

            # target_batch = reward_batch + \
            #                discounted_factor * np.invert(done_batch).astype(float) * (np.amax(q_val_batch, axis=1))

            ## Double Q Network
            # q_val_batch_max_idx = np.argmax(q_val_batch,axis = 1)
            #
            # q_val_target_batch = target_net.model_predict(sess, next_state_batch)
            #
            # target_batch = reward_batch + \
            #                discounted_factor * np.invert(done_batch).astype(float) * \
            #                (q_val_target_batch[np.arange(batch_size), q_val_batch_max_idx] )





            state_batch = np.array(state_batch)

            local_loss = q_eval_net.update_model(sess, state_batch, action_batch, target_batch)

            loss = np.append(loss, np.array(local_loss))

            if (done):
                print ('loss_mean: {:.4E}, max: {:.4E}, min: {:.4E}'.format(\
                    Decimal(np.nanmean(loss)), Decimal(np.nanmax(loss)), Decimal(np.nanmin(loss))) )

                for ti in range(len(transition)):
                    if len(replay_memory) == replay_memory_size:
                        replay_memory.pop(0)
                    replay_memory.append(transition[ti])
                    total_step_ += 1


                break

            state = next_state
            total_step += 1

            if (i_episode+1)<=expert_demo_num_episodes and (i_episode+1)%5==0 and t >= max_iter_num and i_episode>1:
                print ('loss_mean: {:.4E}, max: {:.4E}, min: {:.4E}'.format( \
                    Decimal(np.nanmean(loss)), Decimal(np.nanmax(loss)), Decimal(np.nanmin(loss))))
                break

            if t >= max_iter_num and t % max_iter_num == 0:
                state = env.reset()
                # Stack 4 successive frames for POMDP
                state = np.stack([state] * 4, axis=2)
                transition = []
                local_loss = None
                loss = np.NaN

        if (i_episode+1)==expert_demo_num_episodes:
            print("\nEnd of expert demonstration.\n")

        episode_reward = -len(transition)+ int(done)*reward
        episode_summary.value.add(simple_value=episode_reward, tag='episode/reward')
        episode_summary.value.add(simple_value=t, tag='episode/episode_total_steps')
        episode_summary.value.add(simple_value=total_step_, tag='episode/total_effective_length')
        q_eval_net.summary_writer.add_summary(episode_summary, i_episode+1)
        q_eval_net.summary_writer.flush()



#=======================
# Main Program
#=======================

tf.reset_default_graph()
experiment_dir = os.path.abspath('./experiments/ImitationDRL_v2')


# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Environment Initialization
state = env.reset()

action = np.random.randint(4,size = 1)

q_eval_net= Q_Network(scope = 'q_eval_net', summary_dir0=experiment_dir)

target_net = Q_Network(scope = 'target_net')

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        Q_learning(sess, env, q_eval_net, target_net, num_episodes = 1000, replay_memory_size = 100000,\
                   replay_memory_initial_size = 10000, target_net_update_interval = 10, discounted_factor = 0.99, \
                   epsilon_s = 0.3, epsilon_f = 0.1, batch_size = 32, max_iter_num = 500, expert_demo_num_episodes = 1000)


# tensorboard --logdir='/home/cougarnet.uh.edu/lhuang28/SwarmDRL/Prototype/experiments/ImitationDRL/summary_q_eval_net'  --port 6006