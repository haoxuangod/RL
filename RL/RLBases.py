from __future__ import annotations
import copy
import math
import os
import pickle
import shutil
from collections import deque

import gymnasium.spaces
import numpy as np
import pygame
import torch
from gymnasium.wrappers import RecordEpisodeStatistics

from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
from torch.utils.tensorboard import SummaryWriter

from Model.Model import IterationModel, ModelData
from RL.common.exploration.strategies.continuous import ContinuousExplorationStrategy, GaussianNoise
from RL.common.exploration.strategies.discrete import DiscreteExplorationStrategy, Greedy
from RL.common.replaybuffer import ReplayBuffer, ExperienceReplayBuffer
from RL.common.utils.TBLogCollector import TBLogCollector, write_logs_to_dir
from RL.common.utils.decorator.Meta import SerializeMeta, from_dict
from RL.common.utils.decorator.TorchMeta import TBMeta, SerializeAndTBMeta
from RL.common.wrappers.CSVLoggerWrapper import CSVLoggerWrapper

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set


@dataclass
class Patch:
    # owner 是被修改的容器对象；key 是字段名/索引；old 是原值
    owner: Any
    key: Any
    old: Any


def _is_unpicklable(x: Any) -> bool:
    # 延迟导入，避免没装 pygame 时出错
    try:
        import pygame
        return isinstance(x, (pygame.surface.Surface, pygame.time.Clock))
    except Exception:
        return False


def sanitize_for_pickle(
        root: Any,
        *,
        replace_with=None,
        max_depth: int = 50,
) -> List[Patch]:
    """
    递归遍历 root，把 pygame.Surface 等不可pickle对象替换成 replace_with。
    返回 patches，用于 restore。
    """
    patches: List[Patch] = []
    seen: Set[int] = set()

    def walk(obj: Any, depth: int):
        if depth > max_depth:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        # 1) 容器类型
        if isinstance(obj, dict):
            # dict value
            for k, v in list(obj.items()):
                if _is_unpicklable(v):
                    patches.append(Patch(obj, k, v))
                    obj[k] = replace_with
                else:
                    walk(v, depth + 1)
            # dict key（一般不需要，但保险起见）
            for k in list(obj.keys()):
                walk(k, depth + 1)
            return

        if isinstance(obj, list):
            for i, v in enumerate(list(obj)):
                if _is_unpicklable(v):
                    patches.append(Patch(obj, i, v))
                    obj[i] = replace_with
                else:
                    walk(v, depth + 1)
            return

        if isinstance(obj, tuple):
            # tuple 不可变：如果里面有 surface，没法原地改
            # 一般 env 里 surface 不会放在 tuple；若真遇到，建议在外层容器替换整个 tuple
            for v in obj:
                walk(v, depth + 1)
            return

        if isinstance(obj, set):
            # set 元素不可原地替换：遇到 surface 只能整体重建 set
            if any(_is_unpicklable(v) for v in obj):
                old = obj.copy()
                new = set(replace_with if _is_unpicklable(v) else v for v in obj)
                patches.append(Patch({"__set_owner__": obj}, "__set__", old))
                obj.clear()
                obj.update(new)
            else:
                for v in list(obj):
                    walk(v, depth + 1)
            return

        # 2) 普通对象：遍历 __dict__
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for k, v in list(d.items()):
                if _is_unpicklable(v):
                    patches.append(Patch(d, k, v))  # 注意：owner 这里用 obj.__dict__
                    d[k] = replace_with
                else:
                    walk(v, depth + 1)

        # 3) 常见 wrapper 链：env / unwrapped / etc.
        for attr in ("env", "unwrapped"):
            if hasattr(obj, attr):
                try:
                    walk(getattr(obj, attr), depth + 1)
                except Exception:
                    pass

    walk(root, 0)
    return patches


def restore_after_pickle(patches: List[Patch]) -> None:
    """按 patches 把替换过的值恢复回去。"""
    for p in reversed(patches):
        # set 的特殊 patch
        if isinstance(p.owner, dict) and "__set_owner__" in p.owner and p.key == "__set__":
            s = p.owner["__set_owner__"]
            s.clear()
            s.update(p.old)
        else:
            p.owner[p.key] = p.old


'''
state、next_state均为np.array和gym所获取到的一致
'''


class GameWindow:

    def __init__(self, screen_width=800, screen_height=600, rewards_window_length=200):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # 渲染给定rewards数组最后的若干条
        self.rewards_window_length = rewards_window_length
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))  # 自定义窗口大小
        self.font = pygame.font.Font(None, 20)  # 不扫描系统字体，绕开 win32 字体枚举 bug

    def render_window(self, env, episode, reward, total_reward, action, steps, rewards):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        game_window = True
        try:
            frame = env.render()
        except Exception as e:
            game_window = False
        # time.sleep(0.02)
        if game_window:
            # 渲染到自定义窗口
            self.screen.fill((0, 0, 0))
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                self.screen.blit(pygame.transform.scale(frame_surface, (400, 300)), (0, 0))  # 缩放游戏画面
        self.render_info(episode, reward, total_reward, action, steps, (402, 50))  # 右侧显示
        self.draw_reward_curve(rewards)
        pygame.display.flip()

    def draw_reward_curve(self, rewards):
        fig, ax = plt.subplots(figsize=(5, 3))
        # 强制x轴仅显示整数刻度
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(rewards) < self.rewards_window_length:
            ax.plot(rewards, 'r-', label='reward')
        else:
            ax.plot([i for i in range(len(rewards) - self.rewards_window_length + 1, len(rewards) + 1)],
                    rewards[-self.rewards_window_length:], 'r-', label='reward')

        ax.set_title('Reward Curve')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        ax.legend()
        # 将 Matplotlib 图表转换为 Pygame Surface
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        raw_data = canvas.buffer_rgba().tobytes()  # 获取 RGBA 数据
        size = canvas.get_width_height()

        surf = pygame.image.fromstring(raw_data, size, "RGBA")
        scaled_surf = pygame.transform.scale(surf, (400, 300))
        self.screen.blit(scaled_surf, (0, 301))
        plt.close(fig)

    def render_info(self, episode, current_reward, total_reward, action, steps, position=(10, 10)):
        text_episode = self.font.render(f"Episode:  {episode}", True, (255, 255, 0))
        # 当前奖励（红色）与累计奖励（绿色）
        text_current = self.font.render(f"Current:  {current_reward:.1f}", True, (255, 0, 0))
        text_total = self.font.render(f"Total:  {total_reward:.1f}", True, (0, 255, 0))
        text_action = self.font.render(f"Action: {action}", True, (0, 0, 255))
        text_steps = self.font.render(f"steps: {steps}", True, (255, 255, 255))
        self.screen.blit(text_episode, position)
        self.screen.blit(text_current, (position[0], position[1] + 30))
        self.screen.blit(text_total, (position[0], position[1] + 60))
        self.screen.blit(text_action, (position[0], position[1] + 90))
        self.screen.blit(text_steps, (position[0], position[1] + 120))


class EnvModelData(ModelData):
    pass


class RLInnerModelCallBack:
    def onEpisodeStart(self, episode, model, is_eval=False):
        pass

    def onEpisodeFinished(self, episode, total_reward, model, env, is_eval=False):
        pass

    def onStepFinished(self, episode, state, next_state, action, reward, done, info, model, is_eval=False):
        pass

    def onUpdateFinished(self, update_cnt, model, is_eval=False):
        pass


class StateProcessor(metaclass=SerializeMeta):
    '''
    用于加工原始状态，如标准化某些数据
    '''

    def process(self, state):
        return state

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)


class NormalizeStateProcessor(StateProcessor):
    '''
    单环境（非向量化）
    自动对所有的属性标准化
    需要修改
    '''

    def __init__(self, state_shape, epsilon: float = 1e-8):
        self.count = 0
        self.state_shape = state_shape
        self.mean = np.zeros(self.state_shape, dtype=np.float64)
        self.M2 = np.zeros(self.state_shape, dtype=np.float64)
        self.epsilon = epsilon

    def _update_stats(self, obs: np.ndarray):
        self.count += 1
        delta = obs - self.mean
        self.mean += delta / self.count
        delta2 = obs - self.mean
        self.M2 += delta * delta2

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        var = self.M2 / (self.count - 1) if self.count > 1 else self.M2
        std = np.sqrt(var) + self.epsilon
        return (obs - self.mean) / std

    def process(self, state: np.ndarray) -> np.ndarray:
        # 这个方法会在 step() 后被调用
        self._update_stats(state)
        return self._normalize(state)


# 代理内部与环境互动的模型
class RLInnerModelBase(metaclass=SerializeAndTBMeta):
    '''
    memory记录和更新所需要的columns
    Agent采集原始数据/HRLInnerModel更新节点内部Node,然后再更新Node内的InnerModel会将如下表项添加到memory中：
    Agent:["state","action","reward","next_state","done","length"]每次采样的length=1
    HRLInnerModel中的Node要求:["state","action","reward","next_state","done","length"]这里的length
    可能是1也可能是非原子任务为任意执行长度
    columns需要是其子集才能被其使用
    '''
    columns = ["state", "action", "reward", "next_state", "done", "length"]
    # 默认 "action": np.float32真的没问题吗?
    columns_type_dict = {"state": np.float32, "action": np.float32, "reward": np.float32,
                         "next_state": np.float32, "done": np.float32, "length": np.int32}
    # 支持的State的类型如spaces.Discrete(n)、spaces.Box(low, high, shape)
    support_state_types = []
    # 支持的action的类型如spaces.Discrete(n)、spaces.Box(low, high, shape)
    support_action_types = []

    def __init__(self, state_space, action_space, vec_env_num=None,
                 state_processor: StateProcessor = None, exploration_strategy=None):
        self.steps_done = 0
        self.episode = 0
        self.current_episode_steps_done = 0
        self.state_space = state_space
        self.action_space = action_space
        self._set_action_shape()
        self.callbacks = []
        self.best_reward = -math.inf
        self.best_model = {}
        self.vec_env_num = vec_env_num
        if vec_env_num is not None:
            if not isinstance(vec_env_num, int):
                raise ValueError("vec_env_num must be None or an positive integer")
            if vec_env_num <= 0:
                raise ValueError("vec_env_num must be None or an positive integer")
        if not self.check_state_and_action():
            raise ValueError(f"The type of state_space must in {self.support_state_types} and the type of action "
                             f"type must in {self.support_action_types} but got the class of state_action="
                             f"{self.state_space.__class__} action_space={self.action_space.__class__}")

        self.check_and_set_exploration_strategy(exploration_strategy)
        if state_processor is None:
            self.state_processor = StateProcessor()
        else:
            self.state_processor = state_processor

    def _set_action_shape(self):
        self.action_shape = None

    def _setup_tensorboard(self, tensorboard_log_dir=None, base_tag=None):
        if self.exploration_strategy.tensorboard_log_dir is None:
            tensorboard_log_dir = os.path.join(self.tensorboard_log_dir, self.exploration_strategy.__class__.__name__)
            base_tag = self.base_tag + "/" + self.exploration_strategy.__class__.__name__
            self.exploration_strategy.setup_tensorboard(tensorboard_log_dir, base_tag)

    def check_and_set_exploration_strategy(self, exploration_strategy):
        if isinstance(self.action_space, (int, gymnasium.spaces.Discrete, gymnasium.spaces.Discrete)):
            if exploration_strategy is None:
                self.exploration_strategy = self.get_default_discrete_exploration_strategy()
            else:
                if not isinstance(exploration_strategy, DiscreteExplorationStrategy):
                    raise ValueError(f"The action of the model is {self.action_space.__class__} so the exploration "
                                     f"strategy must be an instance of DiscreteExplorationStrategy but got {exploration_strategy.__class__}")
                self.exploration_strategy = exploration_strategy
        elif isinstance(self.action_space, (gymnasium.spaces.Box)):

            if exploration_strategy is None:
                self.exploration_strategy = self.get_default_continuous_exploration_strategy()
            else:
                if not isinstance(exploration_strategy, ContinuousExplorationStrategy):
                    raise ValueError(f"The action of the model is {self.action_space.__class__} so the exploration "
                                     f"strategy must be an instance of ContinuousExplorationStrategy but got {exploration_strategy.__class__}")
                self.exploration_strategy = exploration_strategy

    def get_default_continuous_exploration_strategy(self):
        # 有问题
        sigma = (self.action_space.high - self.action_space.low) / 2
        return GaussianNoise(sigma=sigma)

    def get_default_discrete_exploration_strategy(self):
        return Greedy()

    def load_best_model(self):
        pass

    def update_best_model(self, env):
        '''
        #怎么能这么写有问题
        self.best_model=self.get_best_model()
        dates = env.date_memory
        values = env.asset_memory
        # 2. 构造 DataFrame
        import pandas as pd
        self.best_model_account_df= pd.DataFrame({
            "date": pd.to_datetime(dates),
            "account_value": values
        })
        '''

    def get_best_model(self):
        pass

    def select_random_action(self):
        return self.action_space.sample()

    def check_state_and_action(self):
        '''
        判断当前state_space、action_space是否满足当前类的要求
        '''
        flag = ((self.state_space.__class__ in self.support_state_types) and
                (self.action_space.__class__ in self.support_action_types))
        return flag

    def update(self, *args, **kwargs):
        # 每个step结束后更新模型
        self._update(*args, **kwargs)

    def _update(self, *args, **kwargs):
        raise NotImplementedError

    def select_action(self, state, is_eval=False):
        state = self.state_processor(state)
        return self._select_action(state, is_eval=is_eval)

    def _select_action(self, state, is_eval=False):
        '''
        :param state: 这里的state已经是经过处理后的state
        '''
        args = self.predict(state)
        if not isinstance(args, tuple):
            action = self.exploration_strategy.select_action(args, step=self.steps_done + 1, is_eval=is_eval,
                                                             episode=self.episode)
        else:
            action = self.exploration_strategy.select_action(*args, step=self.steps_done + 1, is_eval=is_eval,
                                                             episode=self.episode)
        return action

    '''
    def convert_data(self,key,value):
        
        :param key: data中的键如“state”、“action”...
        :param value: data[key]
        :return: 转换为合适的模式如将int/ndarray转换为Torch.tensor
        
        return value
    '''

    def process_state(self, state):
        return self.state_processor(state)

    def predict(self, state):
        '''
        将当前状态输入模型中得到模型的预测值
        后续将作为_select_action的输入
        :param state:
        :return:
        '''
        raise NotImplementedError

    def get_max_action_value(self, states, to_basic_type: bool):
        '''
        获取当前状态下各个动作的最优评分
        :return:
        '''
        raise NotImplementedError

    def get_best_action(self, states, to_basic_type: bool):
        '''
        返回当前状态下的最优动作
        '''
        raise NotImplementedError

    def onEpisodeStart(self, episode, is_eval=False):
        self.episode = episode
        self.current_episode_steps_done = 0
        self._onEpisodeStart(episode, is_eval=is_eval)
        for callback in self.callbacks:
            callback.onEpisodeStart(episode, self, is_eval=is_eval)

    def _onEpisodeStart(self, episode, is_eval=False):
        pass

    def _onEpisodeFinished(self, episode, total_reward, env, is_eval=False):
        pass

    def onEpisodeFinished(self, episode, total_reward, env, is_eval=False):
        self._onEpisodeFinished(episode, total_reward, env)
        if is_eval:
            if total_reward > self.best_reward:
                self.update_best_model(env)
                self.best_reward = total_reward
        '''
        else:
            if total_reward>self.best_reward:
                self.update_best_model()
                self.best_reward = total_reward
        '''
        for callback in self.callbacks:
            callback.onEpisodeFinished(episode, total_reward, self, env, is_eval=is_eval)
        if self.exploration_strategy is not None:
            self.exploration_strategy.reset()

    def _onStepFinished(self, episode, state, next_state, action, reward, done, info, is_eval=False):
        pass

    def onStepFinished(self, episode, state, next_state, action, reward, done, info, is_eval=False):
        self.steps_done = self.steps_done + 1
        self.current_episode_steps_done = self.current_episode_steps_done + 1
        self._onStepFinished(episode, state, next_state, action, reward, done, info, is_eval=is_eval)
        for callback in self.callbacks:
            callback.onStepFinished(episode, state, next_state, action, reward, done, info, self, is_eval=is_eval)


class OnPolicyInnerModel(RLInnerModelBase):
    def update(self, data):
        self._update(data)

    def _update(self, data):
        raise NotImplementedError


class OffPolicyInnerModel(RLInnerModelBase):
    def __init__(self, state_space, action_space, vec_env_num=None, memory: ReplayBuffer = None,
                 gamma=0.99, batch_size=32, memory_capacity=10000, update_freq=4,
                 steps_before_update=1, tensorboard_log_dir=None, exploration_strategy=None, state_processor=None):
        super().__init__(state_space=state_space, action_space=action_space, vec_env_num=vec_env_num,
                         tensorboard_log_dir=tensorboard_log_dir, state_processor=state_processor,
                         exploration_strategy=exploration_strategy)
        # 收集多少步的数据后才开始更新
        self.steps_before_update = steps_before_update
        self.update_freq = update_freq
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.update_cnt = 0
        if steps_before_update > self.memory_capacity:
            raise ValueError("steps_before_update must be less than or equal to memory_capacity")
        self.batch_size = batch_size
        if memory is None:
            self.set_default_memory()
        else:
            self.memory = memory

    def set_default_memory(self):
        info_dict = self.columns_type_dict.copy()
        dic = {"state": self.state_space.shape,
               "action": self.action_space.shape,
               "next_state": self.state_space.shape,
               "reward": (1,),
               "done": (1,),
               "length": (1,)}
        info_dict = {key: (val, dic[key]) for key, val in info_dict.items()}
        self.memory = ExperienceReplayBuffer(capacity=self.memory_capacity, n_steps=1, gamma=0.99,
                                             columns=self.__class__.columns, columns_info_dict=info_dict)

    def append_memory(self, data, episode):
        self.memory.append(data, episode, self)

    def update(self):
        if self.steps_done % self.update_freq == 0 and self.steps_done >= self.steps_before_update:
            if self.memory.can_sample(self.batch_size):

                indices, batch_data, weights = self.memory.sample(self.batch_size)
                data = {column: torch.from_numpy(batch_data[column]).to(self.device) for column in
                        self.__class__.columns}
                D = len(self.state_space.shape)

                state = data["state"]

                if state.ndim == D + 2:
                    # 向量环境
                    b, n, *s = state.shape
                    # state = state.reshape(b * n, *s)
                    data = {column: data[column].reshape(b * n, -1) for column, val in data.items()}

                elif state.ndim == D + 1:
                    # 普通环境
                    pass
                else:
                    raise ValueError(f"Unexpected states shape: {state.shape}")
                self.update_cnt = self.update_cnt + 1
                self._update(indices, data, weights)
                for callback in self.callbacks:
                    callback.onUpdateFinished(self.update_cnt, self)

    def _update(self, indices, batch_data, weights):
        raise NotImplementedError


class MultiStepInnerModelDecorator:
    @classmethod
    def Decorator(cls_decorator, cls):
        class NewClass(cls):
            def __init__(self, *args, n_steps, **kwargs):
                self.n_steps = n_steps
                super().__init__(*args, **kwargs)

        return NewClass

    def __new__(cls_decorator, cls):
        return cls_decorator.Decorator(cls)

    def set_default_memory(self):
        info_dict = self.columns_type_dict.copy()
        dic = {"state": self.state_space.shape,
               "action": self.action_space.shape,
               "next_state": self.state_space.shape,
               "reward": (1,),
               "done": (1,),
               "length": (1,)}
        info_dict = {key: (val, dic[key]) for key, val in info_dict.items()}
        self.memory = ExperienceReplayBuffer(capacity=self.memory_capacity, n_steps=self.n_steps, gamma=0.99,
                                             columns=self.__class__.columns, columns_info_dict=info_dict)


class RLTimeSeriesInnerModelDecorator:

    @classmethod
    def Decorator(cls_decorator, cls):
        class NewClass(cls):
            def __init__(self, *args, window_size, **kwargs):
                self.window_size = window_size
                super().__init__(*args, **kwargs)
                self.state_window = deque(maxlen=self.window_size)
                self.next_state_window = deque(maxlen=self.window_size)
                self.current_state_window = deque(maxlen=self.window_size - 1)
                self.current_episode = 0

            def append_memory(self, data, episode):
                if len(self.state_window) < self.window_size:
                    self.state_window.append(data["state"])
                    self.next_state_window.append(data["next_state"])
                else:
                    self.state_window.popleft()
                    self.state_window.append(data["state"])
                    self.next_state_window.popleft()
                    self.next_state_window.append(data["next_state"])
                if len(self.state_window) == self.window_size:
                    state = np.stack(self.state_window, axis=0)
                    next_state = np.stack(self.next_state_window, axis=0)
                    data1 = copy.deepcopy(data)
                    data1["state"] = state
                    data1["next_state"] = next_state
                    self.memory.append(data1, episode, self)

            def _select_action(self, state, is_eval=False):
                if len(self.current_state_window) < self.window_size - 1:
                    return self.select_random_action()
                else:
                    self.current_state_window.append(state)
                    state1 = np.stack(self.current_state_window, axis=0)
                    self.current_state_window.pop()
                    return super().select_action(state1)

            def _onStepFinished(self, episode, state, next_state, action, reward, done, info):
                super()._onStepFinished(episode, state, next_state, action, reward, done, info)
                if len(self.current_state_window) < self.window_size - 1:
                    self.current_state_window.append(state)
                else:
                    self.current_state_window.popleft()
                    self.current_state_window.append(state)

            def _onEpisodeFinished(self, episode, state):
                super()._onEpisodeFinished(episode, state)
                self.state_window.clear()
                self.next_state_window.clear()
                self.current_state_window.clear()

        return NewClass

    def __new__(cls_decorator, cls):
        return cls_decorator.Decorator(cls)


class RLAgent(IterationModel, metaclass=TBMeta):
    # 交互时记录的数据项
    columns = ['state', 'action', 'reward', 'next_state', 'done', "length"]

    def __init__(self, name,
                 env,
                 inner_model: RLInnerModelBase,
                 per_episode_max_steps=5000,
                 render_mode=0,
                 max_iter=1000,
                 print_interval=10,
                 save_interval=100,
                 directory='', save_name='',
                 monitor_csv: str = None,
                 info_keywords=(),
                 seed=None,
                 verbose: int = 1,
                 running_window: int = 100):
        '''
        render_mode:
            render_mode=0 渲染每一步
            render_mode=1 只在一轮结束后渲染
            其余值不渲染
        '''

        self.seed = seed
        self.epsilon = 1.0
        self.render_mode = render_mode
        self.per_episode_max_steps = per_episode_max_steps
        # Logging controls
        self.verbose = verbose
        self.running_rewards = deque(maxlen=running_window)
        self.running_window = running_window
        if not set(inner_model.columns).issubset(set(self.columns)):
            raise ValueError('columns in InnerModel must be the subset of columns in the Agent')
        self.inner_model = inner_model

        if self.inner_model.tensorboard_log_dir == None and self.tensorboard_log_dir is not None:
            path = os.path.join(self.tensorboard_log_dir, self.inner_model.__class__.__name__)
            base_tag = self.base_tag + "/" + self.inner_model.__class__.__name__
            self.inner_model.setup_tensorboard(path, base_tag=base_tag)

        # Monitor or CSV logger 向量环境?
        if isinstance(env, gymnasium.vector.VectorEnv):
            # wrapper后续考虑怎么加上去
            self.env = env
        else:
            if isinstance(env, gymnasium.Env):
                if monitor_csv:
                    self.env = CSVLoggerWrapper(env, filename=monitor_csv, info_keywords=info_keywords)
                else:
                    self.env = RecordEpisodeStatistics(env)
            else:
                raise TypeError(f"Expected env to be a `gymnasium.Env` but got {type(env)}")
        data = EnvModelData(name=env.__class__.__name__, filepath=None)
        super().__init__(name, data, max_iter, print_interval, save_interval, directory, save_name)
        self.game_window = GameWindow(screen_width=800, screen_height=600, rewards_window_length=200)
        self.eval_reward_list = []

    def log(self, tag: str, value: float, step: int = None):
        step = step if step is not None else self.current_iteration
        if hasattr(self, 'summary_writer') and self.summary_writer:
            self.summary_writer.add_scalar(f"{self.base_tag}/{tag}", value, step)
        if self.verbose >= 1:
            print(f"[{tag}] {value:.4f} @ iter {step}")

    def _setup_tensorboard(self, tensorboard_log_dir=None, base_tag=None):
        if self.tensorboard_log_dir is not None:
            self.tb_log_collector = TBLogCollector(log_root=self.tensorboard_log_dir)
        else:
            self.tb_log_collector = None

    def load_state(self):
        super().load_state()
        tensorboard_log = self.state.parameters["tensorboard_log"]
        if self.tensorboard_log_dir is not None:
            if os.path.exists(self.tensorboard_log_dir):
                try:
                    shutil.rmtree(self.tensorboard_log_dir)
                    print(f"已递归删除目录及其内容：{self.tensorboard_log_dir}")
                except Exception as e:
                    print(f"删除失败：{e}")
            write_logs_to_dir(tensorboard_log, self.tensorboard_log_dir)

        self.inner_model = from_dict(self.state.parameters["inner_model"])
        env = self.state.parameters["env"]
        if isinstance(env, dict):
            self.env = from_dict(env)
        else:
            self.env = env
        # 刷新summary_writer
        self.setup_tensorboard(tensorboard_log_dir=self.tensorboard_log_dir, base_tag=self.base_tag)

    def update_state(self):
        """
        更新模型状态，保存当前优化器的状态到 ModelState。
        """
        super().update_state()
        log = {}
        if self.tensorboard_log_dir is not None:
            log = self.tb_log_collector.collect()
        env = self.env.to_dict() if hasattr(self.env, "to_dict") else self.env
        patches = sanitize_for_pickle(env, replace_with=None)

        try:
            state = {
                "inner_model": self.inner_model.to_dict(),
                "tensorboard_log": log,
                "env": copy.deepcopy(env)
            }
            # print("update_state:",state)
            # 更新 ModelState 的参数
            self.state.update_parameters(state)
        except Exception as e:
            print(f"RLAgent:更新优化器状态时出错: {e}")
        finally:
            restore_after_pickle(patches)
        # HierarchicalMAXQAgent._reset()

    def train_iteration(self, iteration, eval_env=None):
        if eval_env is None:
            train_reward = self.execute_episode(self.env, self.render_mode, is_eval=False)
            if self.tensorboard_log_dir:
                self.summary_writer.add_scalar(self.base_tag + "/reward/train_reward", train_reward,
                                               self.current_iteration)
            return train_reward
        else:
            train_reward = self.execute_episode(self.env, self.render_mode, is_eval=False)
            eval_reward = self.predict(eval_env, -1)
            if self.tensorboard_log_dir:
                self.summary_writer.add_scalar(self.base_tag + "/reward/train_reward", train_reward,
                                               self.current_iteration)
                self.summary_writer.add_scalar(self.base_tag + "/reward/eval_reward", eval_reward,
                                               self.current_iteration)
            self.eval_reward_list.append(eval_reward)
            print("eval_reward:", eval_reward)
            return train_reward

    def execute_episode(self, env, render_mode=0, is_eval=False):
        # 新增对向量环境的支持，直接展平，运行结束后恢复即可
        state, info = env.reset()
        total_reward = 0
        steps = 0
        self.inner_model.onEpisodeStart(episode=self.current_iteration + 1, is_eval=is_eval)
        while steps < self.per_episode_max_steps:
            steps = steps + 1
            action = self.inner_model.select_action(state, is_eval=is_eval)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if not isinstance(env, gymnasium.vector.VectorEnv):
                # 非向量环境
                done = terminated or truncated  # 合并终止标志
            else:
                done = np.logical_or(terminated, truncated)
            self.save_experience(state, action, reward, next_state, done)
            self.inner_model.onStepFinished(self.current_iteration + 1, self.inner_model.process_state(state),
                                            self.inner_model.process_state(next_state), action, reward, done,
                                            info, is_eval=is_eval)
            if not is_eval:
                if isinstance(self.inner_model, OffPolicyInnerModel):
                    self.inner_model.update()
                elif isinstance(self.inner_model, OnPolicyInnerModel):
                    data = {"state": self.inner_model.process_state(state), "action": action, "reward": reward,
                            "next_state": self.inner_model.process_state(next_state),
                            "done": done, "length": 1}
                    self.inner_model.update(data)
            state = next_state
            if render_mode == 0:
                self.game_window.render_window(env, self.current_iteration + 1, reward, total_reward, action, steps,
                                               self.history)
            if not isinstance(env, gymnasium.vector.VectorEnv):
                if done:
                    break
            else:
                if done.all():
                    break
        if isinstance(env, gymnasium.vector.VectorEnv):
            total_reward = total_reward.mean().item()
        total_reward = float(total_reward)
        if render_mode == 1:
            # self.history会少一个因为运行时不会调用保存的代码
            self.game_window.render_window(env, self.current_iteration + 1, 0, total_reward, -1, -1, self.history)
        self.inner_model.onEpisodeFinished(self.current_iteration + 1, total_reward, env=env, is_eval=is_eval)
        # 更新滑动窗口
        self.running_rewards.append(total_reward)
        # 从 info['episode'] 获取统计（若包装 RecordEpisodeStatistics）
        ep = info.get('episode', {})
        ep_r = ep.get('r', total_reward)
        ep_l = ep.get('l', steps)
        ep_t = ep.get('t', None)
        # 记录到 TensorBoard 和打印
        self.log('episode_reward', ep_r)
        self.log('episode_length', ep_l)
        if ep_t is not None:
            self.log('episode_time', ep_t)

        # 记录滑动窗口平均
        if len(self.running_rewards) > 0:
            avg_rw = sum(self.running_rewards) / len(self.running_rewards)
            self.log(f'avg{self.running_window}_reward', avg_rw)

        if self.verbose >= 1:
            extra = f" | avg{self.running_window}={avg_rw:.3f}" if len(self.running_rewards) > 0 else ''
            if is_eval:
                print(f"Prediction done | R={ep_r:.3f} | L={ep_l}{extra}")
            else:
                print(f"Episode {self.current_iteration + 1} done | R={ep_r:.3f} | L={ep_l}{extra}")
        return total_reward

    def load_best_model(self):
        self.inner_model.load_best_model()

    def predict(self, env=None, render_mode=0):
        if env is None:
            env = self.env
        return self.execute_episode(env, render_mode=render_mode, is_eval=True)

    def save_experience(self, state, action, reward, next_state, done):

        data = {"state": self.inner_model.process_state(state),
                "action": action,
                "reward": reward,
                "next_state": self.inner_model.process_state(next_state),
                "done": done,
                "length": 1}

        if isinstance(self.inner_model, OffPolicyInnerModel):
            self.inner_model.append_memory(data, self.current_iteration + 1)

    def draw_history(self, save_path=None):
        plt.figure(figsize=(5, 3))
        plt.plot(self.history, 'r-', label='reward')
        plt.title('Reward Curve')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)
