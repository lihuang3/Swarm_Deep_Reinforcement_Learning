'''============================================================================

File: main.py
From project: pyMaze - Maze generator and game in Python
Date: 2015-03-29
Author: olehermanse ( http://www.github.com/olehermanse )
License: The MIT License (MIT)

The MIT License (MIT)

Copyright (c) 2015 olehermanse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

============================================================================'''

# DISCLAIMER: Unoptimized and uncommented - Proceed at your own risk.

# =================INIT====================
import pygame, sys, os.path, time, numpy as np, random

filename = '/home/lihuang/SwarmDRL/Prototype/Env/MapData/'
mazeData = np.loadtxt(filename + 'map1.txt').astype(int)
obsX, obsY = np.nonzero(np.ones([mazeData.shape[0],mazeData.shape[1]])-mazeData)
freeX, freeY = np.nonzero(mazeData)
sys.setrecursionlimit(10000)
from pygame.locals import *

pygame.init()
fpsClock = pygame.time.Clock()

# =================VARIABLES====================
# Constants
mazeWidth = mazeData.shape[0] #22
mazeHeight = mazeData.shape[1] #18
u = pixelUnit = 50
windowWidth = mazeWidth * pixelUnit
windowHeight = mazeHeight * pixelUnit

# Maze is a 2D array of integers
maze = np.ones([mazeData.shape[0],mazeData.shape[1]])-mazeData

# Maze variables - used for generation
level = 1


# Corner variables - gameplay
checks = 0
goal =  np.array(zip(freeX,freeY)[np.random.choice(len(freeX))])


# Player variables
score = 0
robot_num = 5
# robot_id = random.sample(range(freeX.shape[0]), robot_num)
# playerx = np.zeros([len(robot_id),1]).astype(int)
# playery = np.copy(playerx).astype(int)
# for i in range(len(robot_id)):
#     playerx[i], playery[i] = freeX[robot_id[i]],freeY[robot_id[i]]

frames = 0
fps = 14
highScore = 0
oldScore = 0
playerName = 'P1'

# Pygame window
windowSurfaceObj = pygame.display.set_mode((windowWidth, windowHeight))
updateRect = pygame.Rect(0, 0, u, u)

# Color variables
whiteColor = pygame.Color(255, 255, 255)
blackColor = pygame.Color(0, 0, 0)
redColor = pygame.Color(255, 0, 0)
greenColor = pygame.Color(0, 255, 0)


def minit():
    global level, seed, seedPlus
    level = 1
    seed = seedPlus = 0
    global score, scorePerLevel, highScore, oldScore
    score = scorePerLevel = highScore = oldScore = 0
    global minutes, seconds, secondsLevel, secondsTotal, secondsAverage
    minutes = seconds = secondsLevel = secondsTotal = secondsAverage = 0
    global frames
    frames = 0


# Draw a square with color c at (x,y) in our grid of squares with u width
def drawSquare(x, y, c):
    global u
    pygame.draw.rect(windowSurfaceObj, c, (x * u, y * u, u, u))


# Draw maze walls without player or objectives
def drawMaze():
    for x in range(0, mazeWidth):
        for y in range(0, mazeHeight):
            if maze[x, y] == 1:
                drawSquare(x, y, blackColor)


def updateText():
    global score, level, scorePerLevel, playerName
    global minutes, seconds, frames, secondsLevel, secondsAverage
    levelmsg = ' Level:' + str(level) + '(' + str(seedPlus) + ')'
    scoremsg = ' Score:' + str(score)
    stepmsg = ' || Total steps:' + str(-score+checks*100)

    msg = scoremsg + stepmsg

    pygame.display.set_caption(msg)


# Draw maze, objectives and player. Update score display
def drawScene():
    global minutes, seconds, frames, secondsLevel, secondsTotal
    frames += 1
    if (frames >= fps):
        seconds += 1
        secondsLevel += 1
        secondsTotal += 1
        frames = 0
    if (seconds >= 60):
        minutes += 1
        seconds = 0
    updateText()
    windowSurfaceObj.fill(whiteColor)
    for i in range(playerx.shape[0]):
        drawSquare(playerx[i], playery[i], redColor)
    drawMaze()

    if checks == 0:
        drawSquare(goal[0],goal[1], greenColor)
    pygame.display.update()


# Check if game world coordinate is outside of maze
def isOutside(x, y):
    if x < 0 or y < 0 or x >= mazeWidth or y >= mazeHeight:
        return True
    return False


# Check if game world coordinate is on the edge of the maze
def isBorder(x, y):
    if x == 0 and (y >= 0 and y < mazeHeight):
        return True
    if x == (mazeWidth - 1) and (y >= 0 and y < mazeHeight):
        return True
    if y == 0 and (x >= 0 and x < mazeWidth):
        return True
    if y == mazeHeight - 1 and (x >= 0 and x < mazeWidth):
        return True
    return False


# Check if a game world coordinate is blocked by wall
def isBlocked(x, y):
    if (x < 0 or y < 0 or x >= mazeWidth or y >= mazeHeight):
        return True
    if (maze[int(x), int(y)] == 1):
        return True
    return False




# Tests whether we want to generate a wall at (x,y) based on 2 factors
def cellGen(x, y):
    drawSquare(x, y, blackColor)
    pygame.display.update()
    return


# resets player position and the level
def resetPlayer():
    global checks, playerx, playery, secondsLevel
    robot_id = random.sample(range(freeX.shape[0]), robot_num)
    playerx = np.zeros([len(robot_id), 1]).astype(int)
    playery = np.copy(playerx).astype(int)
    for i in range(len(robot_id)):
        playerx[i], playery[i] = freeX[robot_id[i]], freeY[robot_id[i]]

    checks = 0

    global kUp, kLeft, kDown, kRight
    global kW, kA, kS, kD
    kUp = kLeft = kDown = kRight = False
    kW = kA = kS = kD = False


# Generate a maze based on seed and level
def generate():
    resetPlayer()
    drawScene()


# Moves player by (x*unit, y*unit)
# All game logic is done through this function
# since nothing happens when standing still
def playerMove(x, y):
    global playerx, playery, score
    global level, secondsLevel, minutes, seconds
    global checks, goal
    checkAgg = 1


    for i in range(playerx.shape[0]):
        playerx[i] += int(x)
        playery[i] += int(y)
        if (isBlocked(playerx[i], playery[i])):
            playerx[i] -= x
            playery[i] -= y
        if (goal[0] == playerx[i] and goal[1] ==  playery[i]):
            checkAgg *= 1
        else:
            checkAgg *= 0


    if (checkAgg and checks == 0):
        checks = 1
        score += 100
        return
    elif (checks == 0):
        score -= 1


# Move player based on keyboard input
def movement():
    if kW or kUp:
        playerMove(0, -1)
    if kA or kLeft:
        playerMove(-1, 0)
    if kS or kDown:
        playerMove(0, 1)
    if kD or kRight:
        playerMove(1, 0)


def exitPyMaze():
    pygame.quit()
    sys.exit()



def pad(s, n):
    while (len(s) < n):
        s = s + ' '
    return s





def restart():
    minit()
    if os.path.isfile('save.txt'):
        os.remove('save.txt')
    generate()



# Main:
minit()

generate()

while True:
    # Handle events:
    events = 0
    time.sleep(0.2)

    for event in pygame.event.get():

        events += 1

        if event.type == pygame.QUIT:
            exitPyMaze()
        # Movement is done once per key down event
        # As well as once per frame if key is held down.
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                kW = True
                movement()
            if event.key == pygame.K_a:
                kA = True
                movement()
            if event.key == pygame.K_s:
                kS = True
                movement()
            if event.key == pygame.K_d:
                kD = True
                movement()
            if event.key == pygame.K_UP:
                kUp = True
                movement()
            if event.key == pygame.K_LEFT:
                kLeft = True
                movement()
            if event.key == pygame.K_DOWN:
                kDown = True
                movement()
            if event.key == pygame.K_RIGHT:
                kRight = True
                movement()
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                kW = False
            if event.key == pygame.K_a:
                kA = False
            if event.key == pygame.K_s:
                kS = False
            if event.key == pygame.K_d:
                kD = False
            if event.key == pygame.K_UP:
                kUp = False
            if event.key == pygame.K_LEFT:
                kLeft = False
            if event.key == pygame.K_DOWN:
                kDown = False
            if event.key == pygame.K_RIGHT:
                kRight = False
    # Drawing scene and updating window:
    if (events == 0):
        movement()  # All game logic is done through movement function
    drawScene()
    fpsClock.tick(fps)