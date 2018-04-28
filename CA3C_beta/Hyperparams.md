#### Hyperparameters

- __[`env.reward`](RGBEnv_v1.py#L103):__ (path: RGBEnv_v1.py/MazeEnv/step)     
```buildoutcfg
goal_range = 50
if cost_to_go <= goal_range * self.robot_num:
    done = True
    reward = 100.0
else:
    done = False
    reward = 0
```   
Defualt `goal_range=15`
On 04/10/18, `goal_range=30`


- __[`intrisic reward`](worker.py#L160)__: (path: worker.py/Worker/run_n_step/)
```buildoutcfg
extrinsic_reward = reward
intrinsic_reward = self.local_model.intrinsic_reward(self.state, next_state, action_onehot)
reward += 100*intrinsic_reward
```


- __[`t_max`](train.py/#L33)__: (path: train.py) 
```buildoutcfg
tf.flags.DEFINE_integer("t_max", 5, "Number of steps before performing an update")   
```
Default `t_max=25`
On 04/10/18, `t_max=10`
On 04/12/18, `t_max=20`

- __[`(global) feature_space`](train.py/#L85)__: (path: train.py)
```buildoutcfg
with tf.variable_scope("global"):
    global_net = cnn_lstm(feature_space=512, action_space=4)
    with tf.variable_scope("predictor"):
      global_model = fwd_inv_model(feature_space=512, action_space=4)    
```

- __[`(local) feature_space`](worker.py/#L88)__: (path: worker.py/Worker/init)
```buildoutcfg
with tf.variable_scope(name):
    global_net = cnn_lstm(feature_space=512, action_space=4)
    with tf.variable_scope("predictor"):
      global_model = fwd_inv_model(feature_space=512, action_space=4)    
```


- __[`cnn layers`](estimators2.py/#L18)__: (path: estimator2.py/build_shared_network)
```buildoutcfg
# Three convolutional layers
conv1 = tf.layers.conv2d(
    inputs=X, filters=32, kernel_size=5, strides=3, activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(
    inputs=conv1, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu, name="conv2")
conv3 = tf.layers.conv2d(
    inputs=conv2, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu, name="conv3")

# Fully connected layer
fc1 = tf.layers.dense(
    inputs=tf.contrib.layers.flatten(conv3), units=1024, name="fc1", activation=tf.nn.relu)
```

- __[`A3C optimizer`](estimators2.py/#L150)__: (path: estimator2.py/cnn_lstm/init)
```buildoutcfg
self.optimizer = tf.train.RMSPropOptimizer(0.00005, 0.99, 0.0, 1e-8)
```
Default `learning_rate=0.00025`
On 04/10/18, `learning_rate=0.00001`
On 04/12/18, `learning_rate=0.00008` Failed
On 04/12/18, `learning_rate=0.000025` Better
On 04/16/18, `learning_rate=0.00002` 
- __[`model optimizer`](estimators2.py/#L220)__: (path: estimator2.py/fwd_inv_model/init)
```buildoutcfg
self.optimizer = tf.train.RMSPropOptimizer(0.00005, 0.99, 0.0, 1e-8)
```
Default `learning_rate=0.00025`
On 04/10/18, `learning_rate=0.00001`
On 04/12/18, `learning_rate=0.00008` Failed
On 04/12/18, `learning_rate=0.000025` Better
On 04/16/18, `learning_rate=0.00002` 


_ __[`state`](estimators2.py#L71)__:
```buildoutcfg
self.state = X = tf.placeholder(shape=[None, 84, 84, 1], dtype=tf.uint8, name="X")

```
Default `shape = [None, 84, 84, 4]`
