import numpy as np
import torch


class VanDerPolEnv:
    '''
    任务概述
    VanDerPolEnv 是一个基于范德波尔（Van der Pol）振荡方程的连续控制任务。该系统具有非线性阻尼特性，
    能够产生自激振荡。智能体的目标是在一定时间内通过施加连续控制信号（外部输入）稳定系统状态（例如将状态拉回原点），
    同时尽可能减少控制能量的开销。范德波尔系统常用于测试强化学习在非线性、不稳定动态系统中的控制能力。

    状态（Observation）空间
        类型：Box
        形状：(3,)
        数据类型：float32
        含义：长度为 3 的向量 [t, x, y]，分别表示
            1. 当前时间 t
            2. 系统状态变量 x
            3. 系统状态变量 y（通常为 dx/dt）
        取值范围：无限（Box(-inf, +inf, (3,), float32)）

    动作（Action）空间
        类型：Box
        形状：(1,)
        数据类型：float32
        取值范围：[-1.0, +1.0]
        含义：长度为 1 的连续控制量 u，用作外部输入（施加在系统上的外力）
            目标是通过该输入稳定系统或将其引导至期望轨迹
    '''
    def __init__(self, initial_state=np.array([0, 1, 0]), action_min=np.array([-1]), action_max=np.array([+1]),
                 terminal_time=11, dt=0.1, inner_step_n=10):
        self.state_dim = 3
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.beta = 0.05
        self.r = 0.05
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.initial_state = initial_state
        self.state = self.initial_state

    def reset(self):
        self.state = self.initial_state
        return self.state

    def g(self, state):
        return torch.stack([torch.zeros(state.shape[1]), torch.ones(state.shape[1])]).transpose(0, 1).unsqueeze(1).type(torch.FloatTensor)

    def set_dt(self, dt):
        self.dt = dt
        self.inner_dt = dt / self.inner_step_n

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            f = np.array([1, self.state[2], (1 - self.state[1] ** 2) * self.state[2] - self.state[1] + action[0]])
            self.state = self.state + f * self.inner_dt

        if self.state[0] < self.terminal_time:
            done = False
            reward = - self.r * action[0] ** 2 * self.dt
        else:
            done = True
            reward = - self.state[1] ** 2 - self.state[2] ** 2

        return self.state, reward, done, None

    def get_state_obs(self):
        return 'time: %.3f  x: %.3f y: %.3f' % (self.state[0], self.state[1], self.state[2])

    def render(self):
        print(self.get_state_obs())