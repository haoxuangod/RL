import gymnasium
from gymnasium import spaces
import numpy as np


class PendulumEnv(gymnasium.Env):
    '''
    任务概述
        PendulumEnv 是一个经典的倒立摆控制任务，智能体控制一个单摆杆，通过施加连续的力矩来使其
        尽量在有限时间内保持直立（θ=0）并且角速度尽量为零。环境模拟真实的物理动力学，包括重力、质
        量和杆长，智能体需要最小化角度误差、角速度和控制成本。

    状态（Observation）空间
        类型：Box
        形状：(3,)
        数据类型：float32
        含义：长度为 3 的向量 [t, θ, ω]，分别表示
            1. 当前时间 t
            2. 摆杆角度 θ（以 π 为初始，单位：弧度）
            3. 角速度 ω
        取值范围：无限（Box(-inf, +inf, (3,), float32)）

    动作（Action）空间
        类型：Box
        形状：(1,)
        数据类型：float32
        取值范围：[-2.0, +2.0]
        含义：施加于摆杆的力矩（控制转动速度和方向）
    '''
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 initial_state=np.array([0, np.pi, 0], dtype=np.float32),
                 dt=0.2,
                 terminal_time=5.0,
                 inner_step_n=2,
                 action_min=np.array([-2.0], dtype=np.float32),
                 action_max=np.array([2.0], dtype=np.float32),
                 render_mode=None):
        super().__init__()

        self.state_dim = 3
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.dt = dt
        self.terminal_time = terminal_time
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n
        self.initial_state = initial_state.astype(np.float32)

        self.gravity = 9.8
        self.r = 0.01
        self.beta = self.r
        self.m = 1.0
        self.l = 1.0
        self.render_mode = render_mode

        self.state = self.initial_state.copy()

        # gym API spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_min, high=self.action_max, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state.copy()
        info = {}
        return self.state.copy(), info

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            angle = self.state[1]
            angular_vel = self.state[2]

            dtheta = angular_vel
            domega = - (3 * self.gravity / (2 * self.l)) * np.sin(angle + np.pi) + \
                     (3.0 / (self.m * self.l ** 2)) * action[0]
            delta = np.array([1.0, dtheta, domega], dtype=np.float32) * self.inner_dt
            self.state = self.state + delta

        if self.state[0] >= self.terminal_time:
            reward = - np.abs(self.state[1]) - 0.1 * np.abs(self.state[2])
            terminated = True
        else:
            reward = - self.r * (action[0] ** 2) * self.dt
            terminated = False

        truncated = False
        info = {}

        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            t, theta, omega = self.state
            print(f"time: {t:.3f}, angle: {theta:.3f}, angular velocity: {omega:.3f}")

    def set_dt(self, dt):
        self.dt = dt
        self.inner_dt = dt / self.inner_step_n
