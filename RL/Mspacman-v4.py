'''
在OpenAI Gym的MsPacman-v4环境中，state（观测空间）和action（动作空间）的定义如下：

一、State（观测空间）定义
数据格式
观测空间是Box类型，表示图像像素数据13。
默认输出为210×160像素的RGB三通道图像（形状：(210, 160, 3)），取值范围为[0, 255]。
若使用gym.wrappers 预处理（如灰度化或缩放），可能输出不同格式（例如(84, 84, 1)的灰度图）。
帧堆叠机制
MsPacman-v4默认启用帧堆叠（Frame Stacking），会将连续4帧图像堆叠作为观测值（shape=(4, 210, 160, 3)），以捕捉动态信息。
环境初始化与重置
调用env.reset() 返回初始观测值（即游戏画面第一帧）。
env.step(action) 返回的observation是下一帧的像素数据。
二、Action（动作空间）定义
动作类型
动作空间是Discrete类型，共9个离散动作，对应游戏手柄的8个方向键和“无操作”：
0: 无操作
1: 上
2: 右
3: 左
4: 下
5: 右上（上+右）
6: 左上（上+左）
7: 右下（下+右）
8: 左下（下+左）
动作执行机制
每个动作默认持续4帧（即frame_skip=4），环境会自动重复执行动作以减少计算量。
可通过env.unwrapped.frameskip 参数调整帧跳过次数。


'''

import gymnasium as gym
import torch

from RL.HierarchicalRL.HRLBases import HRLNode, Tree, HRLInnerModel
from RL.HierarchicalRL.HRLModels.HRLDQNModel import HRLDQNInnerModel
from RL.RLBases import EpsilonGreedy, RLAgent
import cv2
import numpy as np
from gymnasium import Wrapper, ObservationWrapper
from collections import deque


class BasicPreprocessWrapper(ObservationWrapper):
    def __init__(self, env, target_size=(84, 84)):
        super().__init__(env)
        self.target_size = target_size
        # 更新观测空间定义
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0,
            shape=(target_size[0], target_size[1], 1),  # 灰度单通道
            dtype=np.float32
        )

    def observation(self, obs):
        # 1. 灰度化 (RGB转单通道)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        # 2. 缩放至目标尺寸 (双线性插值)
        resized = cv2.resize(gray, self.target_size, interpolation=cv2.INTER_AREA)

        # 3. 归一化到[0,1]范围并增加通道维度
        normalized = resized.astype(np.float32) / 255.0
        return np.expand_dims(normalized, axis=-1)


env = gym.make("MsPacman-v4", render_mode="rgb_array")
env = BasicPreprocessWrapper(env)


def HRL():
    # 环境配置
    state_dim = env.observation_space.shape
    print("state_shape:", env.observation_space.shape)
    print("state_type:",type(env.reset()[0]))
    print("action_shape:", env.action_space)
    strategy = EpsilonGreedy()
    #strategy = Boltzmann()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node_inner_model = HRLDQNInnerModel(state_dim,9,device=device ,memory_capacity=50000,n_steps=5,exploration_strategy=strategy, sync_target_strategy=None)
    node=HRLNode("Root",node_inner_model)
    tree=Tree()
    tree.load_from_root(node)
    inner_model = HRLInnerModel(tree=tree,gamma=0.99,exploration_strategy=strategy, memory_capacity=10000,)
    agent = RLAgent("RLAgent", env, inner_model, render_mode=1,directory="Mspacman-v4",save_interval=5)
    #agent.load()
    agent.train()


if __name__ == "__main__":
    HRL()