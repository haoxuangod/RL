import gymnasium
from gymnasium import spaces
import numpy as np
from numpy.linalg import norm


class TargetProblemEnv(gymnasium.Env):
    '''
    任务概述
    TargetEnv 是一个带有目标引导的二维轨迹控制任务。智能体控制一个带质量的小体，在给定时间内通过施加连续二维力（动作），使其尽可能接近目标点 (xG, yG)，同时尽量减少控制能耗。环境会模拟目标参考点、重力和弹性力，并提供稠密奖励，鼓励精确稳定到达目标。

    状态（Observation）空间
        类型：Box
        形状：(7,)
        数据类型：float32
        含义：长度为 7 的向量 [t, x0, y0, x, y, vx, vy]，分别表示
            1. 当前时间 t
            2. 参考点位置 (x0, y0)
            3. 当前质点位置 (x, y)
            4. 当前质点速度 (vx, vy)
        取值范围：无限（Box(-inf, +inf, (7,), float32)）

    动作（Action）空间
        类型：Box
        形状：(2,)
        数据类型：float32
        取值范围：[-1.0, +1.0]²
        含义：二维连续动作 [ux, uy]，分别为 x 和 y 方向的施加力

    '''
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 action_radius=np.array([1.0, 1.0], dtype=np.float32),
                 initial_state=np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                 terminal_time=10.0,
                 dt=0.01,
                 inner_step_n=10,
                 target_point=(2.0, 2.0),
                 render_mode=None):
        super().__init__()

        self.state_dim = 7
        self.action_dim = 2
        self.action_radius = action_radius.astype(np.float32)
        self.action_min = -self.action_radius
        self.action_max = +self.action_radius
        self.dt = dt
        self.inner_step_n = inner_step_n
        self.inner_dt = self.dt / self.inner_step_n
        self.terminal_time = terminal_time
        self.r = 0.001
        self.beta = 0.001
        self.k = 1.0
        self.m = 1.0
        self.g_const = 1.0
        self.xG = target_point[0]
        self.yG = target_point[1]
        self.initial_state = initial_state.astype(np.float32)
        self.render_mode = render_mode

        # gym spaces
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.action_min, high=self.action_max, shape=(2,), dtype=np.float32)

        self.state = self.initial_state.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state.copy()
        info = {}
        return self.state.copy(), info

    def f(self, state, u):
        t, x0, y0, x, y, vx, vy = state
        ux, uy = u
        state_update = np.zeros(self.state_dim, dtype=np.float32)
        state_update[0] = 1.0
        state_update[1] = ux
        state_update[2] = uy
        state_update[3] = vx
        state_update[4] = vy
        state_update[5] = - (self.k / self.m) * (x - x0)
        state_update[6] = - (self.k / self.m) * (y - y0) - self.g_const
        return state_update

    def step(self, action):
        action = np.clip(action, self.action_min, self.action_max)

        for _ in range(self.inner_step_n):
            k1 = self.f(self.state, action)
            k2 = self.f(self.state + k1 * self.inner_dt / 2, action)
            k3 = self.f(self.state + k2 * self.inner_dt / 2, action)
            k4 = self.f(self.state + k3 * self.inner_dt, action)
            self.state = self.state + (k1 + 2 * k2 + 2 * k3 + k4) * self.inner_dt / 6

        t, x0, y0, x, y, vx, vy = self.state
        if t >= self.terminal_time:
            reward = -((x0 ** 2) + (y0 ** 2) + ((x - self.xG) ** 2) + ((y - self.yG) ** 2))
            terminated = True
        else:
            reward = -self.r * norm(action) ** 2 * self.dt
            terminated = False

        truncated = False
        info = {}

        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            t, x0, y0, x, y, vx, vy = self.state
            print(f"x0={x0:.3f}, y0={y0:.3f}, x={x:.3f}, y={y:.3f}, vx={vx:.3f}, vy={vy:.3f}")