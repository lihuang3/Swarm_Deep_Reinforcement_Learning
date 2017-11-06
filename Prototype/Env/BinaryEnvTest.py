import numpy as np
import random
import sys
import time

filename = '/home/lihuang/SwarmDRL/Prototype/Env/MapData/'

mazeData = np.loadtxt(filename + 'map1.csv').astype(int)
costData = np.loadtxt(filename + 'costMap.csv').astype(int)
costData = costData - mazeData



class MazeEnv():
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.maze = np.ones((mazeData.shape[0],mazeData.shape[1]))-mazeData
        self.goal = np.array([1,1])
        self._build_robot()


    def _build_robot(self):
        row, col = np.nonzero(mazeData)
        self.robot_num = 10
        self.robot = random.sample(range(row.shape[0]), self.robot_num)
        self.state = np.zeros(np.shape(mazeData)).astype(int)
        for i in range(self.robot_num):
            self.state[row[self.robot[i]], col[self.robot[i]]] = 10
        output_img = self.state + self.maze * 255
        return output_img

    def step(self,action):
        if action == 0:   # up
            next_direction = 1
            next_axis = 0
        elif action == 1:   # down
            next_direction = -1
            next_axis = 0
        elif action == 2:   # left
            next_direction = -1
            next_axis = 1
        elif action == 3:   # right
            next_direction = 1
            next_axis = 1

        next_state = np.roll(self.state, next_direction, axis=next_axis)

        # Collision check
        collision = np.logical_and(next_state, self.maze)*next_state

        next_state = np.logical_xor(next_state, self.maze)*next_state
        next_state += np.roll(collision, -next_direction, axis=next_axis)

        self.state = next_state
        output_img = self.state + self.maze*255
        cost_to_go = -np.sum(self.state/10*costData)
        if cost_to_go ==0:
            done = True
            reward = 10
        else:
            done = False
            reward = cost_to_go

        return(output_img,reward,done,1)

    def render(self):
        print '\n{}\n'.format(self.state)

    def reset(self):
        return self._build_robot()

# np.random.seed(10)
# env = MazeEnv()
#
# for i in range(100):
#     next_action = np.random.randint(4,size = 1)
#     env.step(next_action)
#     # if i % 100 == 1:

   # sys.stdout.flush()



