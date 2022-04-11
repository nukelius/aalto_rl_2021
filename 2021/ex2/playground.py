import numpy as np
import matplotlib as mpl
from collections import namedtuple
from itertools import product
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon



import numpy as np
from time import sleep
from sailing import SailingGridworld

epsilon = 10e-4  # TODO: Use this criteria for Task 3
gamma = 1

# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)



if __name__ == "__main__":

        done = False
        while not done:
            # Select a random action
            # TODO: Use the policy to take the optimal action (Task 2)
            action = int(np.random.random()*4)

            # Step the environment
            state, reward, done, _ = env.step(action)

            states = []
            probs = []


            for thing in env.transitions[state[0],state[1],action]:
                print(thing.state[0],thing.state[1])
                print(thing.prob)
            done = True
            sleep(0.5)