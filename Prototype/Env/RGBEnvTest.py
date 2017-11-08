
import numpy as np, random, sys, matplotlib.pyplot as plt, time

ROOT_PATH = '/home/lihuang/SwarmDRL/Prototype'

map_data_dir = ROOT_PATH + '/Env/MapData/'

class MazeEnv():
    def __init__(self):
        global mazeData, costData, centerline, mazeHeight, mazeWidth
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        mazeData, costData, centerline = self._load_data(map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape

        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.centerline = np.ones((mazeHeight, mazeWidth))-centerline
        self.goal = np.array([52, 7])
        self._build_robot()

    def _load_data(self, data_directory):
        mazeData = np.loadtxt(data_directory + 'NewMap1.csv').astype(int)
        costData = np.loadtxt(data_directory + 'NewCostMap1.csv').astype(int)
        centerline = np.loadtxt(data_directory + 'CenterlineMap1.csv').astype(int)
        return mazeData, costData, centerline

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

        self.state_img  = np.zeros([mazeHeight,mazeWidth])

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
        plt.imshow(self.state_img + self.maze*255, vmin=0, vmax=255)
        #plt.show()
    def reset(self):
        return self._build_robot()



# np.random.seed(10)
# env = MazeEnv()
# env.render()
# n_epochs = 1000
#
#
# for i in range(n_epochs):
#     next_action = np.random.randint(4,size = 1)
#     state_img,reward, _, _ = env.step(next_action)
#     if i % 100 == 1:
#         plt.subplot( (n_epochs/100+1)/3+1,3, (i/100+1))
#         plt.axis('off')
#         plt.title('Step = ' + str(i) )
#         env.render()
#         #plt.subplots_adjust(wspace = 0.1)
#
# plt.show()
#


