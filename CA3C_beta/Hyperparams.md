#### Hyperparameters

- __`env.reward`:__      
```buildoutcfg
goal_range = 50
if cost_to_go <= goal_range * self.robot_num:
    done = True
    reward = 100.0
else:
    done = False
    reward = 0
```   

- __[`intrisic reward`](worker.py#L145)__:
```buildoutcfg
extrinsic_reward = reward
intrinsic_reward = self.local_model.intrinsic_reward(self.state, next_state, action_onehot)
reward += 1000*intrinsic_reward
```