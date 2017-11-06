import sys
#sys.path.insert(0, '/home/lihuang/SwarmDRL/Prototype/Env')

from Env.BinaryEnvTest import MazeEnv

import time
import numpy as np

env = MazeEnv()

for _ in range(1000):
    env.render()
    env.step(np.random.randint(4,size = 1))
    time.sleep(0.1)


