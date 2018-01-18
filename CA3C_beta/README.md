#### This repo is inherited from dennybritz Github reinforcement-learning
#### Original repo: https://github.com/dennybritz/reinforcement-learning.git

## Implementation of A3C (Asynchronous Advantage Actor-Critic)

#### Running

```
./train.py --model_dir /tmp/a3c --env Breakout-v0 --t_max 5 --eval_every 300 --parallelism 8
```

See `./train.py --help` for a full list of options. Then, monitor training progress in Tensorboard:

```
tensorboard --logdir=/tmp/a3c
```


#### To do list
- [`train.py`](train.py) (1) Re define  ```def make_env(wrap=True)```; (2) play with ```tf.FLAGS```.
- [`worker.py`](worker.py) 



#### Components

- [`train.py`](train.py) contains the main method to start training.
- [`estimators.py`](estimators.py) contains the Tensorflow graph definitions for the Policy and Value networks.
- [`worker.py`](worker.py) contains code that runs in each worker threads.
- [`policy_monitor.py`](policy_monitor.py) contains code that evaluates the policy network by running an episode and saving rewards to Tensorboard.
