"""
A 2D reacher environment.
Adapted from the OpenAI Gym Acrobot environment developed by Christoph Dann
and released under the 3-clause BSD license.
"""

import numpy as np
from numpy import sin, cos, pi
from gym import core, spaces
from gym.utils import seeding
from gym.envs import register
from PIL import ImageColor


class ReacherEnv(core.Env):
    def __init__(self):
        self.viewer = None
        high = np.ones(2) * np.inf
        low = -high
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(5)
        self.state = None
        self.goal = np.array([1., 1.])
        self.termination_threshold = 0.25
        self.seed()
        self.link_length_1 = 1.
        self.link_length_2 = 1.
        self.prev_cartesian_pos = np.zeros(2)
        self.prev_state = np.zeros(2)
        self.step_angle_change = 0.2
        self.substeps = 10

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(2,))
        return self.state

    def get_ee_velocity(self, state, next_state):
        pos1 = self.get_cartesian_pos(state)
        pos2 = self.get_cartesian_pos(next_state)
        delta_pos = pos2 - pos1

        # Let's assume dt is 1...
        return delta_pos

    @property
    def ee_velocity(self):
        return self.get_ee_velocity(self.prev_state, self.state)

    def get_reward(self, prev_state, action, next_state):
        # Just hang around, living a peaceful life of a typical 2D manipulator
        # TODO: Implement me! (either here or in train.py)
        target = np.array([1.,1.])
        max_dist: float = self.link_length_1 + self.link_length_2 + np.sqrt(2)
        dist: float = np.linalg.norm(self.get_cartesian_pos(prev_state)-target)
        return ((max_dist-dist) / max_dist)**0.5


    def get_cartesian_pos(self, state):
        ee_pos = np.zeros(2)
        ee_pos[0] = np.sin(state[0])*self.link_length_1 + \
                np.sin(state[0]+state[1])*self.link_length_2
        ee_pos[1] = -np.cos(state[0])*self.link_length_1 - \
                np.cos(state[0]+state[1])*self.link_length_2
        return ee_pos

    @property
    def cartesian_pos(self):
        return self.get_cartesian_pos(self.state)

    def step(self, a):
        self.prev_cartesian_pos = self.cartesian_pos
        self.prev_state = np.copy(self.state)
        dpos = self.step_angle_change / self.substeps
        joint = a//2
        dpos = dpos*(-1)**a

        # Do the simulation in substeps to avoid a situation where we jump to
        # the other side without terminating the episode
        for _ in range(self.substeps):
            if a < 4:
                self.state[joint] += dpos

        terminal = self.get_terminal_state()

        # Compute the reward
        reward = self.get_reward(self.prev_state, a, self.state)

        return (self.state, reward, terminal, {})

    def get_terminal_state(self):
        terminal_distance = np.sqrt(np.sum((self.cartesian_pos - self.goal)**2))
        terminal = terminal_distance < self.termination_threshold
        return terminal

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500,500)
            bound = self.link_length_1 + self.link_length_2 + 0.2  # 2.2 for default
            self.viewer.set_bounds(-bound,bound,-bound,bound)

        if s is None: return None

        p1 = [-self.link_length_1 *
              cos(s[0]), self.link_length_1 * sin(s[0])]

        p2 = [p1[0] - self.link_length_2 * cos(s[0] + s[1]),
              p1[1] + self.link_length_2 * sin(s[0] + s[1])]

        xys = np.array([[0,0], p1, p2])[:,::-1]
        thetas = [s[0]- pi/2, s[0]+s[1]-pi/2]
        link_lengths = [self.link_length_1, self.link_length_2]

        for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
            l,r,t,b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x,y))
            link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
            link.add_attr(jtransform)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(*[i/256 for i in ImageColor.getrgb("#494947")])
            link.set_color(*[i/256 for i in ImageColor.getrgb("#494947")])
            circ.add_attr(jtransform)

        # Mark the end-effector pos
        jtransform = rendering.Transform(translation=p2[::-1])
        circ = self.viewer.draw_circle(.1)
        circ.set_color(*[i/256 for i in ImageColor.getrgb("#10d46c")])
        circ.add_attr(jtransform)

        # Mark the goal pos
        gtransform = rendering.Transform(translation=self.goal)
        circ = self.viewer.draw_circle(.1)
        circ.set_color(*[i/256 for i in ImageColor.getrgb("#eB5461")])
        circ.add_attr(gtransform)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    env = ReacherEnv()
    for _ in range(10):
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()

register("Reacher-v0",
        entry_point="%s:ReacherEnv"%__name__,
        max_episode_steps=200)

