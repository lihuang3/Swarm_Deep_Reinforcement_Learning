#### This repo is inherited from dennybritz Github reinforcement-learning
#### Original repo: https://github.com/dennybritz/reinforcement-learning.git

## Implementation of Curiosity of A3C (Asynchronous Advantage Actor-Critic)

- For each worker, interacts with enviroment for a few (n) steps
	1. __`run_n_steps`__ In each step, save MDP tuple element `state`, `action`, `next_state`, `reward`, `done` to `Transition` list;
	2. __`update`__ Loop the element in `Transition` list backwards and prepare for the `feed_in` variables, including 
	`states`, `actions`, `value fcn targets`, `policy targets`, `feature encoding $\phi_1$`
	
	



