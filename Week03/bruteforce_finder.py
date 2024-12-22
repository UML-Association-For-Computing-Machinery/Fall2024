#!/usr/bin/python
import math
import random

val = 0

goal = [random.random() * 100, random.random() * 100]

currpath = [random.random() , random.random()]
curr_state = 0
step = [1,1]
reward = 0

prevpath = currpath
currpath[0] = currpath[0] + math.pow(-1, curr_state) * random.random() * step[0]
currpath[1] = currpath[1] + math.pow(-1, curr_state) * random.random() * step[1]
prev_err = [currpath[0] - prevpath[0], currpath[1] - prevpath[1]]
iter = 0

while (currpath != goal):
	if prev_err < curr_state:
		curr_state = 1
		reward = reward - 1
	prevpath = currpath
	prev_err = curr_state
	currpath[0] = currpath[0] + math.pow(-1, curr_state) * random.random() * step[0]
	currpath[1] = currpath[1] + math.pow(-1, curr_state) * random.random() * step[1]
	curr_state = [currpath[0] - prevpath[0], currpath[1] - prevpath[1]]
	print("iteration %d: curr_state: (%0.2lf, %0.2lf), goal_location: (%0.2lf, %0.2lf)", iter, currpath[0], currpath[1], goal[0], goal[1]) 
	iter = iter + 1
	

