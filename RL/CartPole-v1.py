import math
import os

import matplotlib.pyplot as plt

from RL.HierarchicalRL.HRLBases import HRLInnerModel
from RL.HierarchicalRL.HRLModels.HRLDQNModel import HRLDQNInnerModel
from RL.RLBases import RLAgent
from RL.RLModels.ValueOffPolicyInnerModel.DQNModel import DQNInnerModel, DoubleDQNInnerModel
from RL.common.exploration.strategies.discrete import Boltzmann, EpsilonGreedy
import gymnasium as gym

from RL.common.syncTargetNetWorkStrategies.SoftSyncTargetNetWorkStrategy import SoftSyncTargetNetWorkStrategy
from RL.common.syncTargetNetWorkStrategies.StepSyncTargetNetWorkStrategy import StepSyncTargetNetWorkStrategy

'''

CartPole-v1 任务详解
1. 状态空间（State）
    CartPole-v1 的状态由 4 维连续向量 描述，具体参数如下：
    Cart Position：小车在轨道上的水平位置，范围 [-4.8, 4.8]
    Cart Velocity：小车移动速度，无固定范围（理论为 ±∞，实际受物理约束）
    Pole Angle：杆子与垂直方向的夹角，范围 [-24°, 24°]（游戏终止阈值为 ±12°）
    Pole Angular Velocity：杆子的角速度，无固定范围
2. 控制动作（Action）
    动作空间为 离散型，包含两个可选操作：
    action=0：向左施加力（force=-10N）
    action=1：向右施加力（force=+10N）
3. 游戏规则与终止条件
    游戏在以下情况终止：
    杆子倾斜超过阈值：杆子与垂直方向夹角绝对值 >12°
    小车超出轨道范围：水平位置绝对值 >2.4 单位
    达到最大步数：累计存活步数 ≥500（对应奖励上限 500）
4. 奖励机制
    基础奖励：每存活一个时间步（未触发终止条件）奖励 +1
    累计上限：最大总奖励为 500，达到后自动终止
    终止惩罚：游戏结束时 不额外扣除奖励，但最终步的 done=True 可用于训练中的终止信号处理
'''


from RL.HierarchicalRL.MaxQ import *
'''
该任务使用MaxQ没有任何意义
'''
class RootTask(MaxQNode):
    def is_terminal(self,state):
        return False
    '''
    def select_action(self, state, epsilon):
        x, _, theta, _ = state
        if abs(theta) > 0.1:  # 角度偏差较大时优先平衡
            return 0
        elif abs(x) > 1.5:  # 位置偏差较大时优先居中
            return 1
        else:
            return random.choice([0,1])
    '''
class BalancedTask(MaxQNode):
    def is_terminal(self,state):
        theta = abs(state[2])  # 杆子角度绝对值
        # 失败条件：角度超过阈值
        if theta > math.pi/12:
            return True
        '''
        n=5
        # 成功条件（可选）：next_state 连续稳定
        if len(self.memory) >= n:
            last_n = [self.memory[-i][3] for i in range(n)]
            if all([abs(s[2]) < 0.0174 for s in last_n]):
                return True
        return False
        '''
    def shaped_reward(self, state):
        theta = abs(state[2])
        return np.exp(8 * theta)  # 角度越小奖励越高
class PositionTask(MaxQNode):
    def is_terminal(self,state):
        x = abs(state[0])  # 位置绝对值
        # 失败条件：超出安全范围
        if x > 2.4:
            return True
        '''
        n=5
        # 成功条件：回到中心附近
        if len(self.memory) >=n:
            last_n = [self.memory[-i][3] for i in range(n)]
            if all([abs(s[0]) < 0.05 for s in last_n]):
                return True
        '''
        return False

    def shaped_reward(self, state):
        x = abs(state[0])
        return np.exp(2 * x)  # 位置越居中奖励越高


class MoveTask(MaxQNode):
    def is_terminal(self,state):
        return False


def max_Q():
    print(os.getcwd())
    # 环境配置
    env = gym.make('CartPole-v1',render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    tree=MaxQTree()
    # 构建任务树 [1]()
    root = RootTask("Root",state_dim,action_dim=1)

    # 第一层子任务
    balance_task = BalancedTask("Balance",state_dim,action_dim=2)

    position_task = PositionTask("Position", state_dim, action_dim=2)
    #防止代码卡死，Root得做出一个保底决策
    move_task = MoveTask("Move", state_dim,action_dim=2)
    #root.subtasks = [balance_task, position_task,move_task]
    root.subtasks=[move_task]
    '''
    balance_dict = balance_task.to_dict()
    print(balance_dict)
    balance_task = balance_task.from_dict(balance_dict)
    '''
    tree.load_from_root(root)
    '''
    with open("tree_config.json","w") as f:
        dic=tree.config_to_dic()
        print(dic)
        json.dump(dic,f)
    
    tree.save_tree("CartPole-v1_tree.json")
    tree=tree.load_tree("CartPole-v1_tree.json")
    '''
    #tree.load_from_json("tree_config.json")
    print(tree.get_tree_info())
    # 创建Agent
    agent = HierarchicalMAXQAgent(name="HierarchicalMaxQAgent",env=env,tree=tree,print_interval=10,
                                  save_interval=1000,max_iter=50000,sync_target_interval=10,directory="CartPole-v1",save_name="MAXQ")

    #agent.load()
    agent.train()


    '''
    # 训练循环
    for episode in range(1000):
        agent.execute_episode(env)
        # 定期同步目标网络
        if episode % 10 == 0:
            agent.sync_target_network()
    '''

def DQN():
    # 环境配置
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]

    strategy=Boltzmann()
    strategy=EpsilonGreedy(eps_start=0.1,eps_end=0.01,eps_decay=5000)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from tensorboard import program
    tb = program.TensorBoard()
    tensorboard_dir = './tensorboard_record_cartPole-v1/'
    tb.configure(argv=[None, '--logdir', tensorboard_dir, '--port', '6006'])
    url = tb.launch()
    print(f"TensorBoard listening on {url}")

    sync_strategy = SoftSyncTargetNetWorkStrategy(tau=2e-3)
    #sync_strategy = StepSyncTargetNetWorkStrategy(steps_interval=320)
    inner_model = DQNInnerModel(env.observation_space,env.action_space,device=device,
                              memory_capacity=20000, n_steps=5, exploration_strategy=strategy,
                              sync_target_strategy=sync_strategy,lr=0.001,gamma=0.9,
                              steps_before_update=1000, update_freq=10)
    inner_model = DoubleDQNInnerModel(env.observation_space, env.action_space, device=device,
                                memory_capacity=20000, n_steps=5, exploration_strategy=strategy,
                                sync_target_strategy=sync_strategy, lr=0.001, gamma=0.9,
                                steps_before_update=1000, update_freq=10)
    agent=RLAgent("DQNRLAgent", env, inner_model, directory="CartPole-v1",
                    max_iter=20,save_interval=1000, render_mode=1,
                    tensorboard_log_dir=tensorboard_dir)
    #agent.load()
    def callback():
        agent.predict(env, render_mode=0)
    #agent.add_interval_callback((100, callback))
    agent.train()
    #agent.predict(env,render_mode=0)

    draw_history(agent.history,"CartPole-v1-RL.png")
    while 1:
        pass

def draw_history(history,path):
    plt.figure(figsize=(5, 3))
    plt.plot(history, 'r-', label='reward')
    plt.title('Reward Curve')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig(path)
def HRL():
    # 环境配置
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    print("state_type:",type(env.reset()[0]))
    state_dim = env.observation_space.shape[0]
    strategy = EpsilonGreedy()
    #strategy = Boltzmann()
    node_inner_model = HRLDQNInnerModel(state_dim, 2,n_steps=5,exploration_strategy=strategy, sync_target_strategy=None)
    node=HRLNode("Root",node_inner_model)
    tree=Tree()
    tree.load_from_root(node)
    inner_model = HRLInnerModel(tree=tree,gamma=0.99,exploration_strategy=strategy, memory_capacity=10000,)

    agent = RLAgent("RLAgent", env, inner_model, directory="CartPole-v1",save_interval=100,render_mode=1)
    #agent.load()
    agent.train()
    agent.draw_history('CartPole-v1_HRL_EpsilonGreedy_Priority1_nsteps=5')

if __name__ == "__main__":
    #max_Q()
    DQN()
    #HRL()