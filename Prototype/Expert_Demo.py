def expert(self, robot_loc=None):
    transition = []
    if not robot_loc:
        cost_to_goal = costData[robot_loc]
        if cost_to_goal > 1:
            cost_to_goal -= 1
            action = []
            for i in range(4):
                new_pt = robot_loc + np.array([np.cos(i * np.pi / 2), np.sin(i * np.pi / 2)]).astype(int)
                if (costData[new_pt[0], new_pt[1]] == cost_to_goal):
                    if np.absolute(new_pt - robot_loc)[0]:
                        action.append(np.amax([0, (new_pt - robot_loc)[0]]))
                    if np.absolute(new_pt - robot_loc)[1]:
                        action.append(np.amax([2, 2 + (new_pt - robot_loc)[1]]))
                    break
            robot_loc = new_pt

def restart_session(state):
    while (np.sum(state / robot_marker * costData) > self.robot_num):
        robot_loc = np.unravel_index(np.argmax((self.state > 0) * costData), self.state.shape)

        cost_to_goal = costData[robot_loc]
        while cost_to_goal > 1:
            cost_to_goal -= 1
            action = []
            for i in range(4):
                new_pt = robot_loc + np.array([np.cos(i * np.pi / 2), np.sin(i * np.pi / 2)]).astype(int)
                if (costData[new_pt[0], new_pt[1]] == cost_to_goal):
                    if np.absolute(new_pt - robot_loc)[0]:
                        action.append(np.amax([0, (new_pt - robot_loc)[0]]))
                    if np.absolute(new_pt - robot_loc)[1]:
                        action.append(np.amax([2, 2 + (new_pt - robot_loc)[1]]))
                    break
            robot_loc = new_pt

            motion_step += 1
            observation, reward, done, _ = self.step(action[0])
            transition.append([state, action[0], reward, observation, done])
            state = observation

    return motion_step, transition