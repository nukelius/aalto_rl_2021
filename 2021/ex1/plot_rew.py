import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import reacher
import seaborn as sns
import sys
from agent import Agent, Policy
from utils import get_space_dim
from itertools import product

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default=None,
                        help="Policy file to plot")
    parser.add_argument("--env", type=str, default="Reacher-v0",
                        help="Environment to use")
    parser.add_argument("--resolution", type=int, default=101,
                        help="Resolution of the policy/reward image")
    return parser.parse_args(args)

# The main function
def main():
    sns.set()

    args = parse_args()

    # Create a Gym environment
    env = gym.make(args.env)

    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)
    policy = Policy(observation_space_dim, action_space_dim)

    # Load a policy, if passed
    if args.policy:
        policy.load_state_dict(torch.load(args.policy))
        print("Loading policy from", args.policy)
    else:
        print("Plotting a random policy")

    # Create a grid and initialize arrays to store rewards and actions
    npoints = args.resolution
    state_range = np.linspace(-np.pi, np.pi, npoints)
    rewards = np.zeros((npoints, npoints))
    actions = np.zeros((npoints, npoints), dtype=np.int32)

    # Loop through state[0] and state[1]
    for i,th1 in enumerate(state_range):
        for j,th2 in enumerate(state_range):
            # Create the state vector from th1, th2
            state = np.array([th1, th2])

            # Query the policy and find the most probable action
            with torch.no_grad():
                action_dist, _ = policy(torch.from_numpy(state).float().unsqueeze(0))
            action_probs = action_dist.probs.numpy()

            # TODO: What's the best action, according to the policy?
            # Use the action probabilities in the action_probs vector
            # (it's a numpy array)
            actions[i, j] = 0

            # TODO: Compute the reward given state
            rewards[i,j] = 0.

    # Create the reward plot
    num_ticks = 10
    tick_skip = max(1, npoints // num_ticks)
    tick_shift = 2*np.pi/npoints/2
    tick_points = np.arange(npoints)[::tick_skip] + tick_shift
    tick_labels = state_range.round(2)[::tick_skip]
    plt.subplot(1, 2, 1)
    sns.heatmap(rewards)
    plt.xticks(tick_points, tick_labels, rotation=45)
    plt.yticks(tick_points, tick_labels, rotation=45)
    plt.xlabel("J2")
    plt.ylabel("J1")
    plt.title("Reward")

    # Create the policy plot
    plt.subplot(1, 2, 2)
    cmap = sns.color_palette("deep", action_space_dim)
    sns.heatmap(actions, cmap=cmap, vmin=0, vmax=action_space_dim-1)
    plt.xticks(tick_points, tick_labels, rotation=45)
    plt.yticks(tick_points, tick_labels, rotation=45)
    colorbar = plt.gca().collections[0].colorbar
    ticks = np.array(range(action_space_dim))*((action_space_dim-1)/action_space_dim)+0.5
    colorbar.set_ticks(ticks)
    if args.env == "Reacher-v0":
        # In Reacher, we can replace 0..4 with more readable labels
        labels = ["J1+", "J1-", "J2+", "J2-", "Stop"]
    else:
        labels = list(map(str, range(action_space_dim)))
    colorbar.set_ticklabels(labels)
    plt.xlabel("J2")
    plt.ylabel("J1")
    plt.title("Best action")
    plt.suptitle("Rewards and the best action in %s" % args.env)
    plt.show()

if __name__ == "__main__":
    main()
