import gym
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(321)

env = gym.make('CartPole-v0')
env.seed(123)

episodes = 20000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = 2222  # TODO: Set the correct value.
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)


q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q


def find_nearest(array, value):
    return np.argmin(np.abs(array - value))


def get_cell_index(state):
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


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
