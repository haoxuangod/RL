import numpy as np

from RL.common.exploration.strategies.continuous.noises import Noise


class OUNoise(Noise):
    def __init__(self,action_dim,mu=0.0,theta=0.15, sigma=0.2):
        self.mu, self.theta, self.sigma = mu, theta, sigma
        self.state = None
        self.cnt=0
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        self.cnt=0

    def noise(self, action_dim=None,step=0,episode=None):
        x=self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state =x+dx
        self.cnt=self.cnt+1
        return self.state