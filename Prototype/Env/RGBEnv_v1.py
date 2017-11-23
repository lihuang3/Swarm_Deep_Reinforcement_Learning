
import numpy as np, random, sys, matplotlib.pyplot as plt, time, os

random.seed(10)
map_data_dir =os.path.abspath('./MapData')

class MazeEnv():
    def __init__(self):
        global mazeData, costData, centerline, freespace, mazeHeight, mazeWidth, robot_marker, goal_range
        robot_marker = 150
        goal_range = 10
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        mazeData, costData, centerline, freespace = self._load_data(map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape

        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.centerline = np.ones((mazeHeight, mazeWidth))-centerline
        self.freespace = np.ones((mazeHeight, mazeWidth))-freespace
        self.goal = np.array([73, 10])
        self.init_state = []
        self._build_robot()

    def _load_data(self, data_directory):
        mazeData = np.loadtxt(data_directory + '/scaled_maze7.csv').astype(int)
        freespace = np.loadtxt(data_directory + '/scaled_maze7_freespace.csv').astype(int)
        costData = np.loadtxt(data_directory + '/scaled_maze7_costmap.csv').astype(int)
        centerline = np.loadtxt(data_directory + '/scaled_maze7_centerline.csv').astype(int)
        return mazeData, costData, centerline, freespace

    def _build_robot(self):
        row, col = np.nonzero(freespace)

        if not len(self.init_state):
            self.robot_num = 10
            self.robot = random.sample(range(row.shape[0]), self.robot_num)
            self.state = np.zeros(np.shape(mazeData)).astype(int)
            self.state_img = np.copy(self.state)
            for i in range(self.robot_num):
                self.state[row[self.robot[i]], col[self.robot[i]]] += robot_marker
                self.state_img[row[self.robot[i]]-1:row[self.robot[i]]+2,
                    col[self.robot[i]]-1:col[self.robot[i]]+2] = robot_marker*np.ones([3,3])

            self.init_state = self.state
            self.init_state_img = self.state_img
        else:
            self.state = self.init_state
            self.state_img = self.init_state_img


        output_img = self.state_img + self.maze * 255

        return output_img

    def step(self,action):
        if action == 0:   # up
            next_direction = -1
            next_axis = 0
        elif action == 1:   # down
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
        collision = np.logical_and(next_state, self.freespace)*next_state

        next_state *= np.logical_xor(next_state, self.freespace)

        # Move robots in the obstacle area back to previous grids and obtain the next state
        ## Case 1: overlapping with population index
        next_state += np.roll(collision, -next_direction, axis=next_axis)
        ## Case 2: overlapping w/o population index (0: no robot; 1: robot(s) exits)
        # next_state = np.logical_or(np.roll(collision, -next_direction, axis=next_axis), next_state).astype(int)

        # next_state *= robot_marker   # Mark robot with intensity 150

        row, col = np.nonzero(next_state)

        self.state_img  = np.zeros([mazeHeight,mazeWidth])

        for i in range(row.shape[0]):
            self.state_img[row[i]-1:row[i]+2, col[i]-1:col[i]+2] = robot_marker * np.ones([3, 3])

        self.state = next_state

        output_img = self.state_img + self.maze*255

        cost_to_go = np.sum(self.state * costData / robot_marker)
        if cost_to_go <= goal_range * self.robot_num:
            done = True
            reward = 100.0
        else:
            done = False
            reward = -1

        return(output_img,reward,done,1)

    def render(self):
        plt.imshow(self.state_img + self.maze*255, vmin=0, vmax=255)
        plt.show()

    def reset(self):
        return self._build_robot()


    def expert(self, robot_loc):

        if not len(robot_loc):
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
        if (np.sum(self.state*costData/robot_marker)>self.robot_num*goal_range):
            robot_loc = np.unravel_index(np.argmax((self.state > 0) * costData), self.state.shape)
        return self.expert(robot_loc)

# env = MazeEnv()
# env.render()
# n_epochs = 1000
# robot_loc =[]
#
# for i in range(n_epochs):
#     #next_action = np.random.randint(4,size = 1)
#     next_action, robot_loc = env.expert(robot_loc)
#     state_img,reward, done, _ = env.step(next_action)
#     print i, reward, done
#     if i % 100 == 1:
#         print ''
#     #     plt.subplot( (n_epochs/100+1)/3+1,3, (i/100+1))
#     #     plt.axis('off')
#     #     plt.title('Step = ' + str(i) )
#         env.render()
#         plt.show()
#     #     #plt.subplots_adjust(wspace = 0.1)
#
#     if done:
#         env.render()
#         plt.show()
#         print i
#         break


