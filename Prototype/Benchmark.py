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
        self.robot_num = 10
        self._build_robot()

    def _load_data(self, data_directory):
        mazeData = np.loadtxt(data_directory + 'map1.csv').astype(int)
        costData = np.loadtxt(data_directory + 'costMap.csv').astype(int)
        return mazeData, costData


    def _build_robot(self):
        row, col = np.nonzero(mazeData)
        self.robot = random.sample(range(row.shape[0]), self.robot_num)
        self.state = np.zeros(np.shape(mazeData)).astype(int)
        for i in range(self.robot_num):
            self.state[row[self.robot[i]], col[self.robot[i]]] = 10
            # self.state[1,1] = 10
        output_img = self.state + self.maze
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
            #reward = cost_to_go
            reward = -1.0

        return(output_img,reward,done,1)

    def render(self):
        print '\n{}\n'.format(self.state)

    def reset(self, robot_num):
        self.robot_num = robot_num
        return self._build_robot()

    def motion_planning(self):
        motion_step = 0
        while( np.sum(self.state / 10 * costData) > self.robot_num):
            robot_loc = np.unravel_index(np.argmax( (self.state>0) * costData),self.state.shape)

            cost_to_goal = costData[robot_loc]
            while cost_to_goal > 1:
                cost_to_goal -= 1
                action = []
                for i in range(4):
                    new_pt = robot_loc + np.array([np.cos(i * np.pi / 2), np.sin(i * np.pi / 2)]).astype(int)
                    if (costData[new_pt[0],new_pt[1]] == cost_to_goal) :
                        if np.absolute(new_pt-robot_loc)[0]:
                            action.append(np.amax([0,(new_pt-robot_loc)[0]]))
                        if np.absolute(new_pt-robot_loc)[1]:
                            action.append(np.amax([2,2+(new_pt-robot_loc)[1]]))
                        break
                robot_loc = new_pt
                while len(action)>0:
                    motion_step += 1
                    self.step(action[0])
                    action.pop(0)


        return motion_step

# Main Program

np.random.seed(10)

env = MazeEnv()

exp_data = np.zeros([1001,len(np.arange(5,31,5))]).astype(int)
exp_data[0,] = np.arange(5,31,5)

for j in np.arange(5,31,5):
    for i in range(1000):
        env.reset(j)
        total_step = env.motion_planning()
        exp_data[i+1,j/5-1] = total_step
        sys.stdout.flush()
print('Mean: {}'.format(np.mean(exp_data,0)))
print('Std: {}'.format(np.std(exp_data,0)))

np.savetxt('exp_data.txt', exp_data, fmt = '%3d')
