import gym
import numpy as np
from matplotlib import pyplot as plt
from utils import ReplayMemory, Transition
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion


env = gym.make("CartPole-v0")
actions = env.action_space.n


class Agent(object):
    def __init__(self, num_actions, gamma=0.98, memory_size=5000, batch_size=32):
        self.scaler = None
        self.featurizer = None
        self.q_functions = None
        self.gamma = gamma
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.memory = ReplayMemory(memory_size)
        self.initialize_model()

    def initialize_model(self):
        # Draw some samples from the observation range and initialize the scaler
        obs_limit = np.array([4.8, 5, 0.5, 5])
        samples = np.random.uniform(-obs_limit, obs_limit, (1000, obs_limit.shape[0]))
        self.scaler = StandardScaler()
        self.scaler.fit(samples)

        # Initialize the RBF featurizer
        self.featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=80)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=50)),
        ])
        self.featurizer.fit(self.scaler.transform(samples))

        # Create a value approximator for each action
        self.q_functions = [SGDRegressor(learning_rate="constant", max_iter=500, tol=1e-3)
                       for _ in range(self.num_actions)]

        # Initialize it to whatever values; implementation detail
        for q_a in self.q_functions:
            q_a.partial_fit(self.featurize(samples), np.zeros((samples.shape[0],)))

    def featurize(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        # Task 1a: TODO: Implement handcrafted features as (s, abs(s))
        # Task 1b: Use the RBF features using the featurizer
        return self.featurizer.transform(self.scaler.transform(state))

    def get_action(self, state, epsilon=0.0):
        # TODO Task 1: Implement epsilon-greedy
        # Hint: See exercise 3
        raise NotImplementedError("Implement epsilon-greedy")

    def single_update(self, state, action, next_state, reward, done):
        # Calculate feature representations of the
        # Task 1: TODO: Set the feature state and feature next state
        featurized_state = 0
        featurized_next_state = 0

        # Task 1:  TODO Get Q(s', a) for the next state
        next_qs = 0

        # Calculate the updated target Q- values
        # Task 1: TODO: Calculate target based on rewards and next_qs
        target = 0

        # Update Q-value estimation
        self.q_functions[action].partial_fit(featurized_state, target)

    def update_estimator(self):
        if len(self.memory) < self.batch_size:
            # Use the whole memory
            samples = self.memory.memory
        else:
            # Sample some data
            samples = self.memory.sample(self.batch_size)

        # Task 2: TODO: Reformat data in the minibatch
        states = 0
        action = 0
        next_states = 0
        rewards = 0
        dones = 0

        # Task 2: TODO: Calculate Q(s', a)
        featurized_next_states = self.featurize(next_states)
        next_qs = 0

        # Calculate the updated target values
        # Task 2: TODO: Calculate target based on rewards and next_qs
        targets = 0

        # Calculate featurized states
        featurized_states = self.featurize(states)

        # Get new weights for each action separately
        for a in range(self.num_actions):
            # Find states where a was taken
            idx = action == a

            # If a not present in the batch, skip and move to the next action
            if np.any(idx):
                act_states = featurized_states[idx]
                act_targets = targets[idx]

                # Perform a single SGD step on the Q-function params
                self.q_functions[a].partial_fit(act_states, act_targets)

    def store_transition(self, *args):
        self.memory.push(*args)

