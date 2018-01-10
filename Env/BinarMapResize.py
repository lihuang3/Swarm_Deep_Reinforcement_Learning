import numpy as np, os
import matplotlib.pyplot as plt

ROOT_PATH = os.path.abspath('./MapData')

# Load hand-craft binary maze
mazeData = np.loadtxt(ROOT_PATH + '/map1.txt').astype(int)

# Set up scaling ratio
scaling_ratio = 7

# Copy and expand the original maze by the scaling factor
tmp_maze = np.zeros((mazeData.shape[0]*scaling_ratio,mazeData.shape[1]))

scaled_maze = np.zeros((mazeData.shape[0]*scaling_ratio,mazeData.shape[1]*scaling_ratio))

for i in range(tmp_maze.shape[0]):
    tmp_maze[i,:] = mazeData[i/scaling_ratio,:]

for i in range(scaled_maze.shape[1]):
    scaled_maze[:,i] = tmp_maze[:,i/scaling_ratio]

np.savetxt('scaled_maze{}.csv'.format(scaling_ratio), scaled_maze, fmt= '%3d')

# Extract the freespace of the scaled maze (assuming a robot takes 3*3 grids)
scaled_maze *= np.roll(scaled_maze, 1, axis = 0)
scaled_maze *= np.roll(scaled_maze, -1, axis = 0)
scaled_maze *= np.roll(scaled_maze, 1, axis = 1)
scaled_maze *= np.roll(scaled_maze, -1, axis = 1)

np.savetxt('scaled_maze{}_freespace.csv'.format(scaling_ratio), scaled_maze, fmt= '%3d')
# plt.imshow(scaled_maze)
# plt.show()

# Extract centerline of the scaled maze
centerline = np.copy(scaled_maze)
centerline *= np.roll(centerline, 2, axis = 0)
centerline *= np.roll(centerline, -2, axis = 0)
centerline *= np.roll(centerline, 2, axis = 1)
centerline *= np.roll(centerline, -2, axis = 1)

np.savetxt('scaled_maze{}_centerline.csv'.format(scaling_ratio), centerline, fmt = '%3d')


# plt.imshow(Centerline)
# plt.show()

# Object: assign cost-to-go to elements of the centerline
# Method: breadth-first search
BSF_Frontier = []
# Set goal location
goal = np.array([73,10])
# Initialize centerline cost-to-go map
cl_costMap = np.copy(centerline)
BSF_Frontier.append(goal)
cost = 100
cl_costMap[BSF_Frontier[0][0],BSF_Frontier[0][1]] = cost

while len(BSF_Frontier)>0:
    cost = cl_costMap[BSF_Frontier[0][0],BSF_Frontier[0][1]]+1
    for i in range(4):
        new_pt = BSF_Frontier[0]+np.array([np.cos(i*np.pi/2),np.sin(i*np.pi/2)]).astype(int)
        if cl_costMap[new_pt[0],new_pt[1]] == 1.0:
            BSF_Frontier.append(new_pt)
            cl_costMap[new_pt[0], new_pt[1]] = cost
    BSF_Frontier.pop(0)


cl_costMap -= 99*centerline

np.savetxt('scaled_maze{}_clcostmap.csv'.format(scaling_ratio), cl_costMap, fmt = '%3d')

# Object: assign cost-to-go to the whole free space corresponding to the nearest neighbor in the centerline set
# Method: use scipy KD tree to query the nearest neighbor in the centerline
from scipy import spatial
cl_x, cl_y = np.nonzero(centerline)
tree = spatial.KDTree(zip(cl_x.ravel(), cl_y.ravel()))

freespace_pts = np.array(np.nonzero(scaled_maze))
freespace_pts = np.transpose(freespace_pts)

query_result = np.transpose(tree.query(freespace_pts))
tree_idx = query_result[:,1].astype(int)
tree_dist = np.ceil(query_result[:,0])

# Object: assign cost-to-go to elements of the free space

# Initialize free space cost-to-go map
costMap = np.copy(scaled_maze)

for i in range(freespace_pts.shape[0]):
    costMap[freespace_pts[i,0], freespace_pts[i,1]] = cl_costMap[cl_x[tree_idx[i]], cl_y[tree_idx[i]]]+\
                                                       tree_dist[i]

np.savetxt('scaled_maze{}_costmap.csv'.format(scaling_ratio), costMap, fmt = '%3d')



