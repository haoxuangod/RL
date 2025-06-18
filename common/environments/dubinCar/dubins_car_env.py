import gymnasium
import numpy as np
from gymnasium import spaces

class DubinsCarEnv(gymnasium.Env):
    '''
    任务概述
        DubinsCarEnv 是一个模拟 Dubins 车辆的二维最优控制任务。智能体控制一辆只能向前移动、以有限转向速率行驶的小车，
        目标是在总时长 T 内尽量以最小的控制代价完成导航。每个时间步智能体可施加一个连续的角速度动作，以控制小车的朝向，
        环境根据当前位置、角度偏差与控制能耗给予惩罚。最终目标是让小车在 T 时刻达到目标状态。

    状态（Observation）空间
        类型：Box
        形状：(4,)
        数据类型：float32
        含义：长度为 4 的向量 [t, x, y, θ]，分别表示
            1. 当前时间 t（单位：秒）
            2. 小车的水平位置 x
            3. 小车的垂直位置 y
            4. 小车的朝向角 θ（弧度制）
        取值范围：无限（Box(-inf, +inf, (4,), float32)）

    动作（Action）空间
        类型：Box
        形状：(1,)
        数据类型：float32
        取值范围：[-0.5, +1.0]
        含义：长度为 1 的向量 [ω]，表示角速度控制量
            当 ω < 0 时，车辆左转
            当 ω > 0 时，车辆右转
            环境内部将动作进行缩放后用于角度更新
    '''
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,
                 initial_state=np.array([0, 0, 0, 0], dtype=np.float32),
                 dt=0.1,
                 terminal_time=2 * np.pi,
                 inner_step_n=20,
                 action_min=np.array([-1.0], dtype=np.float32),
                 action_max=np.array([1.0], dtype=np.float32),
                 render_mode=None):

        super().__init__()
        self.state_dim = 4
        self.action_dim = 1
        self.action_min = action_min
        self.action_max = action_max
        self.terminal_time = terminal_time
        self.dt = dt
        self.beta = 0.01
        self.r = 0.01
        self.initial_state = np.array(initial_state, dtype=np.float32)
        self.inner_step_n = inner_step_n
        self.inner_dt = dt / inner_step_n
        self.render_mode = render_mode

        # Action and observation space
        self.action_space = spaces.Box(low=action_min, high=action_max, shape=(1,), dtype=np.float32)
        obs_low = np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32)
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.state = self.initial_state.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state.copy()
        info = {}
        return self.state.copy(), info

    def step(self, action):
        action_raw = action.copy()
        action = np.clip(action, self.action_min, self.action_max)
        action = action * 0.75 + 0.25

        for _ in range(self.inner_step_n):
            dx0 = 1.0
            dx1 = np.cos(self.state[3])
            dx2 = np.sin(self.state[3])
            dx3 = action[0]
            delta = np.array([dx0, dx1, dx2, dx3], dtype=np.float32) * self.inner_dt
            self.state = self.state + delta

        x0, x1, x2, x3 = self.state
        if x0 >= self.terminal_time:
            reward = -np.abs(x1 - 4.0) - np.abs(x2) - np.abs(x3 - 0.75 * np.pi)
            terminated = True
        else:
            reward = -self.r * (action_raw[0] ** 2) * self.dt
            terminated = False

        truncated = False
        info = {}

        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"time: {self.state[0]:.3f}  x: {self.state[1]:.3f}  y: {self.state[2]:.3f}  theta: {self.state[3]:.3f}")

    def set_dt(self, dt):
        self.dt = dt
        self.inner_dt = dt / self.inner_step_n