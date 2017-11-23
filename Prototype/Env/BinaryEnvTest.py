import numpy as np, random, sys, os, time, matplotlib.pyplot as plt

map_data_dir = os.path.abspath('./MapData')


class MazeEnv():
    def __init__(self):
        global mazeData, costData, mazeHeight, mazeWidth, robot_marker, goal_range
        goal_range = 10
        self.action_space = ['u', 'd', 'l', 'r']
        robot_marker = 150
        self.n_actions = len(self.action_space)
        mazeData, costData = self._load_data(map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape
        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.goal = np.array([73,10])
        self.init_state = []
        self._build_robot()

    def _load_data(self, data_directory):
        mazeData = np.loadtxt(data_directory + '/scaled_maze7_freespace.csv').astype(int)
        costData = np.loadtxt(data_directory + '/scaled_maze7_costmap.csv').astype(int)
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
        if cost_to_go <= goal_range:
            done = True
            reward = 10.0
        else:
            done = False
            reward = cost_to_go

        return(output_img,reward,done,1)

    def render(self):
        print '\n{}\n'.format(self.state)

    def reset(self):
        return self._build_robot()

    def expert(self, robot_loc):

        if len(robot_loc)==0:
            return self.expert_restart_session()

        cost_to_goal = costData[robot_loc[0], robot_loc[1]]
        if cost_to_goal >1:
            cost_to_goal -= 1

            for i in range(4):
                new_pt = robot_loc + np.array([np.cos(i * np.pi / 2), np.sin(i * np.pi / 2)]).astype(int)
                if (costData[new_pt[0], new_pt[1]] == cost_to_goal):
                    if np.absolute(new_pt - robot_loc)[0]:
                        action = (np.amax([0, (new_pt - robot_loc)[0]]))
                    if np.absolute(new_pt - robot_loc)[1]:
                        action = (np.amax([2, 2 + (new_pt - robot_loc)[1]]))
                    robot_loc = new_pt
                    return action, robot_loc
                elif(costData[new_pt[0], new_pt[1]] == cost_to_goal+1):
                    if np.absolute(new_pt - robot_loc)[0]:
                        action = (np.amax([0, (new_pt - robot_loc)[0]]))
                    if np.absolute(new_pt - robot_loc)[1]:
                        action = (np.amax([2, 2 + (new_pt - robot_loc)[1]]))

                    robot_loc = new_pt
            return action, robot_loc


        else:
            return self.expert_restart_session()



    def expert_restart_session(self):
        if (np.sum(self.state*costData/robot_marker)>goal_range):
            robot_loc = np.unravel_index(np.argmax((self.state > 0) * costData), self.state.shape)
        return self.expert(robot_loc)


# #

env = MazeEnv()
env.reset()
robot_loc = []
for i in range(1000):
    # next_action = np.random.randint(4,size = 1)
    next_action, robot_loc = env.expert(robot_loc)

    state, reward, done, _ = env.step(next_action)
    if i%100 ==0:
        print ' '
    print 'step = {}, action = {}, reward = {}, done = {}'.format(i, next_action, reward, done )
    if done:
        break


   # sys.stdout.flush()



