
import gym
import torch

from RL.HierarchicalRL.HRLBases import HRLNode, Tree, HRLInnerModel
from RL.HierarchicalRL.HRLModels.HRLDQNModel import HRLDQNInnerModel, HRLNAFInnerModel
from RL.RLBases import EpsilonGreedy, RLAgent
import cv2
import numpy as np
from gym import Wrapper, ObservationWrapper
'''
CarRacing-v2 是一个顶视图的连续控制任务，智能体需要通过像素观测来驾驶一辆后轮驱动赛车，目标如下：

最大化累积奖励
    每帧都会被扣除 −0.1 点，以鼓励尽量少用时间完成赛道；
    每访问一个赛道格子可获得 +1000/N 点奖励（N 为本次赛道总格子数），当所有格子都被覆盖时累计可达 1000 点 。
完成赛道
    当所有赛道格子都被访问过后，仿真结束，智能体获得最终累计奖励；

避免离开赛道
    如果汽车跑出赛道外（超出可玩区域），会立刻收到 −100 点惩罚并且仿真终止 。

换句话说，CarRacing-v2 的关键在于“以最快的速度、最少的偏离”覆盖尽可能多的赛道格子，从而在最短时间内拿到最高分 。


CarRacing-v2（默认连续动作）
Observation（状态）空间
类型：Box取值范围：0,255形状：(96,96,3)
数据类型：uint8
含义：每个时间步返回一帧 RGB 图像（96×96 像素），即赛车当前视角的像素观测。 

Action（动作）空间
类型：Box
维度：3
取值范围：
steering ∈ [−1.0,+1.0]
−方向盘转角，−1.0 为最大左转，+1.0 为最大右转

gas ∈ [0.0,1.0]
油门，0.0 无加速，1.0 全油门

brake ∈ 
[0.0,1.0]：刹车，0.0 不刹车，1.0 全刹车
数据类型：float32

表示方式：Box(low=[-1,0,0], high=[1,1,1], shape=(3,), dtype=np.float32) 

可选：离散动作模式（continuous=False）
若创建时传入 gym.make('CarRacing-v2', continuous=False)，则动作空间变为：

类型：Discrete(5)

五个动作 id：
do nothing
steer left
steer right
gas
brake 
'''
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


env = gym.make("CarRacing-v2", render_mode="rgb_array")
env = BasicPreprocessWrapper(env)


def HRL():
    print(env.observation_space)
    # 环境配置
    state_dim = env.observation_space.shape
    print("state_shape:", env.observation_space.shape)
    print("state_type:",type(env.reset()[0]))
    print("action_shape:", env.action_space)
    strategy = EpsilonGreedy()
    #strategy = Boltzmann()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node_inner_model = HRLNAFInnerModel(env.observation_space,env.action_space,device=device ,memory_capacity=50000,n_steps=5,exploration_strategy=strategy, sync_target_strategy=None)
    node=HRLNode("Root",node_inner_model)
    tree=Tree()
    tree.load_from_root(node)
    inner_model = HRLInnerModel(tree=tree,gamma=0.99,exploration_strategy=strategy, memory_capacity=10000,)
    agent = RLAgent("RLAgent", env, inner_model, render_mode=0,directory="CarRacing-v2",save_interval=5)
    #agent.load()
    agent.train()


if __name__ == "__main__":
    HRL()