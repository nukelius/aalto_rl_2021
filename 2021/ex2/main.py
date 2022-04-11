# Copyright 2020 (c) Aalto University - All Rights Reserved
# ELEC-E8125 - Reinforcement Learning Course
# AALTO UNIVERSITY
#############################################################


import numpy as np
from time import sleep
from sailing import SailingGridworld
from tqdm import tqdm

MAX_ITER = 50
NO_EPISODES = 1000
reward = 10
penalty = -10
epsilon = 10e-4
gamma = 0.9
actions = {0: 'LEFT', 1: 'DOWN', 2: 'RIGHT', 3: 'UP'}

# Set up the environment
env = SailingGridworld(rock_penalty=penalty, harbour_reward=reward)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)

if __name__ == "__main__":
    # Reset the environment
    state = env.reset()

    # initialise state values and the policy
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    val_converged = False
    policy_converged = False

    for iter_index in range(MAX_ITER):
        prev_vals = np.copy(value_est)
        prev_policy = np.copy(policy)

        for x in range(env.w):
            for y in range(env.h):

                # if it's a terminal state, the episode ends, but the boat is assumed to remain, so next state is
                # assumed to be the same state - The fixed value of the terminal have been set to zero,
                # but we may also treat it as an absorbing state in which case it only transitions to itself,
                # so it has a fixed value [max_iter * (reward or penalty)]
                if (x, y) == (env.harbour_x, env.harbour_y):
                    value_est[x, y] = MAX_ITER * reward * 0
                elif env.is_rocks(x, y):
                    value_est[x, y] = MAX_ITER * penalty * 0

                # if it is any other state, we use the value iteration update rule
                else:
                    # a dict to store action-value pairs
                    q_vals = {}
                    # for every (non-terminal) state, iterate through all actions
                    for action in range(env.NO_ACTIONS):
                        action_value = 0
                        # value of the action is the weighted probability of
                        # each state transition multiplied by
                        # [immediate reward + discounted value of the next state]
                        for possible_transition in env.transitions[x, y, action]:
                            # calculate expected reward for each possible transition
                            next_x, next_y = possible_transition.state[0], possible_transition.state[1]
                            action_value += (possible_transition.prob *
                                             (possible_transition.reward + gamma * prev_vals[next_x, next_y]))
                        # store all possible action-value pairs
                        q_vals[action] = action_value
                    # apply update rule to the current state
                    value_est[x, y] = max(q_vals.values())
                    # extract greedy policy of the current state alongside each update
                    policy[x, y] = max(q_vals, key=q_vals.get)

        # check if the value iteration algorithm has converged, break if it has
        if np.max(np.abs(value_est - prev_vals)) <= epsilon and not val_converged:
            print("Value iteration converged at iteration number {}".format(iter_index))
            val_converged = True
        # check if the policy converged – usually converges a few iterations ahead of the value iteration
        if np.array_equal(policy, prev_policy) and not policy_converged:
            print("Policy iteration converged at iteration number {}".format(iter_index))
            policy_converged = True
            print(policy_converged)




    sleep(1)

    env.draw_values(value_est)
    env.clear_text()
    env.draw_values(policy)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)

    rewards = np.zeros(NO_EPISODES)

    for episode in tqdm(range(NO_EPISODES), ncols=100, desc="Progress", unit=" episodes"):
        done = False
        state = env.reset()
        while not done:
            action = policy[state[0], state[1]]
            # Step the environment
            state, reward, done, _ = env.step(action)
        rewards[episode] = reward

    mean_rewards, std_dev = np.mean(rewards, dtype=np.float64), np.std(rewards, dtype=np.float64)
    print("average reward: {}, standard deviation: {}".format(mean_rewards, std_dev))
