
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import time
mazeData = np.loadtxt('NewMap1.csv').astype(int)
centerline = np.loadtxt('CenterlineMap1.csv').astype(int)
costData = np.loadtxt('NewCostMap1.csv').astype(int)



class MazeEnv():
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.maze = np.ones((mazeData.shape[0],mazeData.shape[1]))-mazeData
        self.mazeHeight = mazeData.shape[0]
        self.mazeWidth = mazeData.shape[1]
        self.centerline = np.ones((mazeData.shape[0],mazeData.shape[1]))-centerline
        self.goal = np.array([52, 7])
        self._build_robot()


    def _build_robot(self):
        row, col = np.nonzero(centerline)
        idx = np.logical_and(col%5 ==2, row%5==2)
        row = row[idx]
        col = col[idx]
        self.robot_num = 10
        self.robot = random.sample(range(row.shape[0]), self.robot_num)
        self.state = np.zeros(np.shape(mazeData)).astype(int)
        self.state_img = np.copy(self.state)
        for i in range(self.robot_num):
            self.state[row[self.robot[i]], col[self.robot[i]]] = 1
            self.state_img[row[self.robot[i]]-2:row[self.robot[i]]+3, col[self.robot[i]]-2:col[self.robot[i]]+3] = 20*np.ones([5,5])
        output_img = self.state_img + self.maze * 255

        return output_img

    def step(self,action):
        if action == 0:   # up
            next_direction = 5
            next_axis = 0
        elif action == 1:   # down
            next_direction = -5
            next_axis = 0
        elif action == 2:   # left
            next_direction = -5
            next_axis = 1
        elif action == 3:   # right
            next_direction = 5
            next_axis = 1

        next_state = np.roll(self.state, next_direction, axis=next_axis)
        # Collision check
        collision = np.logical_and(next_state, self.centerline)*next_state

        next_state *= np.logical_xor(next_state, self.centerline)

        next_state += np.roll(collision, -next_direction, axis=next_axis)

        row, col = np.nonzero(next_state)

        self.state_img  = np.zeros([self.mazeHeight,self.mazeWidth])

        for i in range(row.shape[0]):
            self.state_img[row[i] - 2:row[i] + 3,col[i] - 2:col[i] + 3] += 20 *next_state[row[i],col[i]]*np.ones([5, 5])

        self.state = next_state

        output_img = self.state_img + self.maze*255

        cost_to_go = -np.sum(self.state*costData)

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

from matplotlib import animation

#np.random.seed(10)
# env = MazeEnv()
#
#
#
# #plt.ion()
# img = plt.imshow(env.state_img,vmin = 0, vmax = 255)
# t0 = time.clock()
# for i in range(500):
#     next_action = np.random.randint(4,size = 1)
#     img,reward, _, _ = env.step(next_action)
#     print reward
#     # if i % 100 == 1:
#     # if i%50 == 0:
#     #     # img.set_data(env.state_img)
#     #     plt.imshow(img, vmin=0, vmax=255)
#     #     plt.show()
#     #     time.sleep(0.5)
#
# print time.clock()
#
# plt.imshow(img, vmin=0, vmax=255)
# plt.show()