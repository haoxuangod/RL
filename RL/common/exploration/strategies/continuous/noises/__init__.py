from abc import ABC, abstractmethod

import numpy as np

from RL.common.schedules.schedules import LinearSchedule


class Noise(ABC):
    def __call__(self, action_dim,step: int,episode:int) -> np.ndarray:
        return self.noise(action_dim,step,episode)
    @abstractmethod
    def noise(self,action_dim,step,episode:int) -> np.ndarray:
        pass

