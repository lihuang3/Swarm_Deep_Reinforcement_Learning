import numpy as np, random, sys, os, time, matplotlib.pyplot as plt

ROOT_PATH = '/home/lihuang/SwarmDRL/Prototype'

map_data_dir = ROOT_PATH + '/Env/MapData/'


class MazeEnv():
    def __init__(self):
        global mazeData, costData, mazeHeight, mazeWidth, robot_marker
        self.action_space = ['u', 'd', 'l', 'r']
        robot_marker = 150
        self.n_actions = len(self.action_space)
        mazeData, costData = self._load_data(map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape
        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.goal = np.array([10,1])
        self.init_state = []
        self._build_robot()

    def _load_data(self, data_directory):
        mazeData = np.loadtxt(data_directory + 'map1.csv').astype(int)
        costData = np.loadtxt(data_directory + 'costMap.csv').astype(int)
        return mazeData, costData


    def _build_robot(self):
        row, col = np.nonzero(mazeData)
        if len(self.init_state)==0:
            self.robot_num = row.shape[0]
            #self.robot = random.sample(range(row.shape[0]), self.robot_num)
            self.state = np.zeros(np.shape(mazeData)).astype(int)
            for i in range(self.robot_num):
               #self.state[row[self.robot[i]], col[self.robot[i]]] = 10
                self.state[row[i], col[i]]= 150

            #self.state[1,1] = 10
            self.init_state = self.state
        else:
            self.state = self.init_state

        output_img = self.state + 255*self.maze


        return output_img

    def step(self,action):
        if action == 0:   # down
            next_direction = -1
            next_axis = 0
        elif action == 1:   # up
            next_direction = 1
            next_axis = 0
        elif action == 2:   # left
            next_direction = -1
            next_axis = 1
        elif action == 3:   # right
            next_direction = 1
            next_axis = 1

        # Translation motion is achieved by np.roll along 'next_axis' dimension, sign(next_direction) direction
        # and abs(next_direction) steps
        next_state = np.roll(self.state, next_direction, axis=next_axis)

        # Collision check: find robots in the obstacle area
        collision = np.logical_and(next_state, self.maze)*next_state

        # Find robots in the free space
        next_state = np.logical_xor(next_state, self.maze)*next_state

        # Move robots in the obstacle area back to previous grids and obtain the next state
        ## Case 1: overlapping with population index
        # next_state += np.roll(collision, -next_direction, axis=next_axis)
        ## Case 2: overlapping w/o population index (0: no robot; 1: robot(s) exits)
        next_state = np.logical_or(np.roll(collision, -next_direction, axis=next_axis), next_state).astype(int)
        next_state *= robot_marker   # Mark robot with intensity 150

        self.state = next_state
        output_img = self.state + 255*self.maze
        cost_to_go = np.sum(self.state*costData/robot_marker)-1
        if cost_to_go == 0:
            done = True
            reward = 10.0
        else:
            done = False
            reward = -1.0

        return(output_img,reward,done,1)

    def render(self):
        print '\n{}\n'.format(self.state)

    def reset(self):
        return self._build_robot()

# # #

# env = MazeEnv()
# env.reset()
#
# for i in range(5000):
#     next_action = np.random.randint(4,size = 1)
#
#     state, reward, done, _ = env.step(next_action)
#
#     print 'action = {}, reward = {}, done = {}'.format(next_action, reward, done )
#     if done:
#         print i
#         break
#

#    # sys.stdout.flush()



