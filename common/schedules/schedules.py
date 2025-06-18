from RL.common.schedules import Schedule


class LinearSchedule(Schedule):
    def __init__(self, total_steps: int, initial_value: float, final_value: float):
        self.total_steps = total_steps
        self.initial_value = initial_value
        self.final_value = final_value

    def value(self, step: int) -> float:
        frac = min(step / self.total_steps, 1.0)
        return self.initial_value + frac * (self.final_value - self.initial_value)

class ExponentialSchedule(Schedule):
    def __init__(self, decay_rate: float, initial_value: float, min_value: float = 0.0):
        self.decay_rate = decay_rate
        self.initial_value = initial_value
        self.min_value = min_value

    def value(self, step: int) -> float:
        val = self.initial_value * (self.decay_rate ** step)
        return max(val, self.min_value)

import math

class CosineSchedule(Schedule):
    def __init__(self, total_steps: int, initial_value: float, final_value: float):
        self.total_steps = total_steps
        self.initial_value = initial_value
        self.final_value = final_value

    def value(self, step: int) -> float:
        step = min(step, self.total_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.total_steps))
        return self.final_value + (self.initial_value - self.final_value) * cosine_decay

class InverseTimeSchedule(Schedule):
    def __init__(self, initial_value: float, decay_rate: float, min_value: float = 0.0):
        self.initial_value = initial_value
        self.decay_rate = decay_rate
        self.min_value = min_value

    def value(self, step: int) -> float:
        val = self.initial_value / (1 + self.decay_rate * step)
        return max(val, self.min_value)
