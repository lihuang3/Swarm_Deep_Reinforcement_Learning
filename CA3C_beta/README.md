#### This repo is inherited from dennybritz Github reinforcement-learning
#### Original repo: https://github.com/dennybritz/reinforcement-learning.git

## Implementation of Curiosity of A3C (Asynchronous Advantage Actor-Critic)

- For each worker, syncs up with the global networks, and interacts with enviroment for a few (n) steps
	1. __`run_n_steps`__ In each step, save MDP tuple element `state`, `action`, `next_state`, `reward`, `done` to `Transition` list;
	2. __`update`__ Loop the element in `Transition` list backwards and prepare for the `feed_in` variables, including 
	`states`, `actions`, `value fcn targets`, `policy targets`, `feature encoding`, and update the neural network

- Training: apply local gradients to the global variables
    1. extract local grads and global vars, return `optimizer.apply_gradients(local_grads + global vars)`	

- Networks
    1. __`CNN`__ _return_ `feature encoding`
    2. __`LSTM`__ _return_ `feature space state`
    3. __`Policy`__ _return_ ``
    4. __`Value Fcn`__
    5. __`Inverse Dynamics`__
    6. __`Forward Dynamics`__
    	
	



