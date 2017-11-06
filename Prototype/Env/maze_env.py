"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


MazeData = np.loadtxt('map1.txt').astype(int)
row,col = np.nonzero(MazeData)
row = np.expand_dims(row,1)
col = np.expand_dims(col,1)
obstacle = np.concatenate((row,col), axis = 1)


UNIT = 40  # pixels
MAZE_H = MazeData.shape[0] # grid height
MAZE_W = MazeData.shape[1]  # grid width
side_len = 18

class MazeEnv(tk.Tk, object):
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        # self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        # for c in range(0, MAZE_W * UNIT, UNIT):
        #     x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
        #     self.canvas.create_line(x0, y0, x1, y1)
        # for r in range(0, MAZE_H * UNIT, UNIT):
        #     x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
        #     self.canvas.create_line(x0, y0, x1, y1)


        hell1_center = UNIT*obstacle
        self.hell1 = np.zeros((hell1_center.shape[0],1)).astype(int)

        for i in range(hell1_center.shape[0]):
            self.hell1[i] = self.canvas.create_rectangle(
                hell1_center[i,0] - side_len, hell1_center[i,1] - side_len,
                hell1_center[i,0] + side_len, hell1_center[i,1] + side_len,
                fill='black')
        # create origin
        origin = np.array([25, 25])
        #
        # # hell
        # hell1_center = origin + np.array([UNIT * 2, UNIT])
        # self.hell1 = self.canvas.create_rectangle(
        #     hell1_center[0] - side_len, hell1_center[1] - side_len,
        #     hell1_center[0] + side_len, hell1_center[1] + side_len,
        #     fill='black')
        # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - side_len, hell2_center[1] - side_len,
        #     hell2_center[0] + side_len, hell2_center[1] + side_len,
        #     fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            hell1_center[45, 0] - side_len, hell1_center[45,1] - side_len,
            hell1_center[45, 0] + side_len, hell1_center[45,1] + side_len,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            hell1_center[0,0] - side_len, hell1_center[0,1] - side_len,
            hell1_center[0,0] + side_len, hell1_center[0,1] + side_len,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - side_len, origin[1] - side_len,
            origin[0] + side_len, origin[1] + side_len,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT


        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 10
            done = True
        # elif next_coords in [self.canvas.coords(self.hell1)]:
        #     reward = -1
        #     done = True
        else:
            reward = -1
            done = False
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()


