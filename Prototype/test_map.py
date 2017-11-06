from maze_env import MazeEnv
import time
import numpy as np
env = MazeEnv()

for _ in range(1000):
    env.render()
    env.step(np.random.randint(4,size = 1))
    time.sleep(0.1)

