import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(123)

env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = 4

# Reasonable values for Cartpole discretization
discr = 16
# x_min, x_max = -2.4, 2.4
# v_min, v_max = -3, 3
# th_min, th_max = -0.3, 0.3
# av_min, av_max = -4, 4


# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]


x_min, x_max = -1.2, 1.2
y_min, y_max = -0.3, 1.2
x_dot_min, x_dot_max = -2.4, 2.4
y_dot_min, y_dot_max = -2, 2
theta_min, theta_max = -6.28, 6.28
theta_dot_min, theta_dot_max = -8, 8
cl_min, cr_min = 0, 0
cl_max, cr_max = 1, 1

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 2222  # TODO: Set the correct value.
initial_q = 0  # T3: Set to 50

# Create discretization grid
# x_grid = np.linspace(x_min, x_max, discr)
# v_grid = np.linspace(v_min, v_max, discr)
# th_grid = np.linspace(th_min, th_max, discr)
# av_grid = np.linspace(av_min, av_max, discr)


x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(y_min, y_max, discr)
x_dot_grid = np.linspace(x_dot_min, x_dot_max, discr)
y_dot_grid = np.linspace(y_dot_min, y_dot_max, discr)
theta_grid = np.linspace(theta_min, theta_max, discr)
theta_dot_grid = np.linspace(theta_dot_min, theta_dot_max, discr)
cl_grid = np.array([cl_min, cl_max])
cr_grid = np.array([cr_min, cr_max])

q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, num_of_actions)) + initial_q


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    # x = find_nearest(x_grid, state[0])
    # v = find_nearest(v_grid, state[1])
    # th = find_nearest(th_grid, state[2])
    # av = find_nearest(av_grid, state[3])
    # return x, v, th, av

    x = find_nearest(x_grid, state[0])
    y = find_nearest(y_grid, state[1])
    x_dot = find_nearest(x_dot_grid, state[2])
    y_dot = find_nearest(y_dot_grid, state[3])
    theta = find_nearest(theta_grid, state[4])
    theta_dot = find_nearest(theta_dot_grid, state[5])
    cl = find_nearest(cl_grid, state[6])
    cr = find_nearest(cr_grid, state[7])
    return x, y, x_dot, y_dot, theta, theta_dot, cl, cr


def get_action(state, q_array, epsilon, greedy=False):
    cell = get_cell_index(state)
    if greedy:
        return np.argmax(q_array[cell])
    else:
        if np.random.random() < epsilon:
            return np.random.randint(num_of_actions)
        else:
            return np.argmax(q_array[cell])


def update_q_value(old_state, action, new_state, reward, done, q_array):
    # TODO: Implement Q-value update
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)
    q_value = q_array[old_cell_index][action]
    q_target = reward + gamma * np.max(q_array[new_cell_index])
    q_array[old_cell_index][action] += alpha * (q_target - q_value)


# Training loop
ep_lengths, epl_avg = [], []

for ep in range(episodes + test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = 2222 / (2222 + ep)
    epsilon_0 = 0
    while not done:
        action = get_action(state, q_grid, epsilon, greedy=False)
        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep - 500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep - 200):])))

# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY

# Calculate the value function
values = q_grid.max(axis=4)  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY

print(q_grid.shape)
print(values)

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()
