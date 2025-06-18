import numpy as np

from RL.common.exploration.strategies.continuous.noises import Noise
class GaussianNoise(Noise):
    def __init__(self, sigma=0.2):
        self.sigma = sigma
    def noise(self, action_dim,step=None,episode=None):
        return np.random.normal(0, self.sigma, size=action_dim)
