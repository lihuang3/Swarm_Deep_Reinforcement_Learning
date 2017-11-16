import numpy as np, random, sys, os, time, matplotlib.pyplot as plt

ROOT_PATH = '/home/lihuang/SwarmDRL/Prototype'

map_data_dir = ROOT_PATH + '/Env/MapData/'


class MazeEnv():
    def __init__(self):
        global mazeData, costData, mazeHeight, mazeWidth
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        mazeData, costData = self._load_data(map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape
        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.goal = np.array([10,1])
        self._build_robot()

    def _load_data(self, data_directory):
        mazeData = np.loadtxt(data_directory + 'map1.csv').astype(int)
        costData = np.loadtxt(data_directory + 'costMap.csv').astype(int)
        return mazeData, costData


    def _build_robot(self):
        row, col = np.nonzero(mazeData)
        self.robot_num = 10
        self.robot = random.sample(range(row.shape[0]), self.robot_num)
        self.state = np.zeros(np.shape(mazeData)).astype(int)
        for i in range(self.robot_num):
           self.state[row[self.robot[i]], col[self.robot[i]]] = 10
           #self.state[1,1] = 10
        output_img = self.state + self.maze
        init_reward = np.sum(self.state/10*costData)-self.robot_num

        return output_img, init_reward

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

        next_state = np.roll(self.state, next_direction, axis=next_axis)

        # Collision check
        collision = np.logical_and(next_state, self.maze)*next_state

        next_state = np.logical_xor(next_state, self.maze)*next_state
        next_state += np.roll(collision, -next_direction, axis=next_axis)

        self.state = next_state
        output_img = self.state + self.maze
        cost_to_go = np.sum(self.state/10*costData)-self.robot_num
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
# np.random.seed(10)
#
# env = MazeEnv()
#
# for i in range(5000):
#     next_action = np.random.randint(4,size = 1)
#     next_action = 2
#     state, reward, done, _ = env.step(next_action)
# #     # if i % 100 == 1:
#     print 'action = {}, reward = {}, done = {}'.format(next_action, reward, done )
#
#
#    # sys.stdout.flush()



