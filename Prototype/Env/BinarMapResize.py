import numpy as np
import matplotlib.pyplot as plt

filename = '/home/lihuang/SwarmDRL/Prototype/Env/MapData/'

mazeData = np.loadtxt(filename + 'map1.txt').astype(int)

scaling_ratio = 1

tmpMaze = np.zeros((mazeData.shape[0]*scaling_ratio,mazeData.shape[1]))
MazeAugment = np.zeros((mazeData.shape[0]*scaling_ratio,mazeData.shape[1]*scaling_ratio))

for i in range(tmpMaze.shape[0]):
    tmpMaze[i,:] = mazeData[i/scaling_ratio,:]

for i in range(MazeAugment.shape[1]):
    MazeAugment[:,i] = tmpMaze[:,i/scaling_ratio]

#
# MazeAugment *= np.roll(MazeAugment, 1, axis = 0)
# MazeAugment *= np.roll(MazeAugment, -1, axis = 0)
# MazeAugment *= np.roll(MazeAugment, 1, axis = 1)
# MazeAugment *= np.roll(MazeAugment, -1, axis = 1)

np.savetxt('map1.csv', MazeAugment, fmt = '%3d')
# plt.imshow(MazeAugment)
# plt.show()

centerline = np.copy(MazeAugment)
# centerline *= np.roll(centerline, 2, axis = 0)
# centerline *= np.roll(centerline, -2, axis = 0)
# centerline *= np.roll(centerline, 2, axis = 1)
# centerline *= np.roll(centerline, -2, axis = 1)
#
#
#
# np.savetxt('CenterlineMap.csv', centerline, fmt = '%3d')


# plt.imshow(Centerline)
# plt.show()

# Breadth-first search for cost-to-go map

BSF_Frontier = []
# goal = np.array([52, 7]).astype(int)
goal = np.array([10,1])
costMap = np.copy(centerline)
BSF_Frontier.append(goal)
cost = 100
costMap[BSF_Frontier[0][0],BSF_Frontier[0][1]] = cost

while len(BSF_Frontier)>0:
    cost = costMap[BSF_Frontier[0][0],BSF_Frontier[0][1]]+1
    for i in range(4):
        new_pt = BSF_Frontier[0]+np.array([np.cos(i*np.pi/2),np.sin(i*np.pi/2)]).astype(int)
        if costMap[new_pt[0],new_pt[1]] == 1.0:
            BSF_Frontier.append(new_pt)
            costMap[new_pt[0], new_pt[1]] = cost
    BSF_Frontier.pop(0)


costMap -= 99*centerline
#costMap /= 5

np.savetxt('costMap.csv', costMap, fmt = '%3d')
