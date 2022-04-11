import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
from time import sleep
import seaborn as sns
import pandas as pd
import sys
import reacher
from agent import Agent, Policy
from utils import get_space_dim


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train-episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Policy update batch size")
    parser.add_argument("--min-update-samples", type=int, default=2000,
                        help="Minimum number of state transitions per update")
    parser.add_argument("--render-training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render-test", action='store_true', help="Render test")
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, render=False, silent=False,
          train_run_id=0, min_update_samples=2000):
    # Arrays to keep track of rewards
    reward_history, timestep_history, updates = [], [], []
    num_updates = 0

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_log_prob = agent.get_action(observation)
            previous_observation = observation.copy()

            # Perform the action on the environment, get new state and reward
            observation, reward, done, info = env.step(action)

            # TODO: Task 1 - change the reward function
            #reward = new_reward(previous_observation, action, observation, env)

            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action, observation,
                    reward, action_log_prob, done)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        updates.append(num_updates)
        timestep_history.append(timesteps)

        # Update the policy, if we have enough data
        if len(agent.states) > min_update_samples:
            agent.update_policy()
            num_updates += 1

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "updates": updates,
                         "reward": reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False):
    total_test_reward, total_test_len = 0, 0
    for ep in range(episodes):
        done = False
        observation = env.reset()
        test_reward, test_len = 0, 0
        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(observation, evaluation=True)
            observation, reward, done, info = env.step(action)
            # TODO: New reward function
            # reward = new_reward(observation)
            if render:
                env.render()
                # Sleep to reduce the fps
                sleep(0.02)
            test_reward += reward
            test_len += 1
        total_test_reward += test_reward
        total_test_len += test_len
        print("Test ep reward:", test_reward)
    print("Average test reward:", test_reward/episodes, "episode length:", test_len/episodes)


# TODO: Definition of the modified reward function
#def new_reward(state, action, next_state, env=None):
#    return 1


# The main function
def main(args):
    sns.set()

    # Create a Gym environment
    env = gym.make(args.env)

    # Exercise 1
    # TODO: For CartPole-v0 - maximum episode length
    env._max_episode_steps =200

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, args.batch_size)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        training_history = train(agent, env, args.train_episodes,
                args.render_training, min_update_samples=args.min_update_samples)

        # Save the model
        model_file = "%s_params.ai" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="updates", y="reward", data=training_history, ci="sd")
        #sns.lineplot(x="episode", y="mean_reward", data=training_history)
        #plt.legend(["Reward", "100-episode average"])
        plt.title("Reward history (%s)" % args.env)
        plt.show()
        print("Training finished.")
    else:
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test(agent, env, args.train_episodes, args.render_test)


# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

