import random
import time

import gym
import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from torch import optim, nn
from torch.distributions import MultivariateNormal

from Decorator.Meta import SerializeMeta
from RL.HierarchicalRL.HRLBases import HRLNodeInnerModelDecorator
from RL.RLBases import RLInnerModel, PrioritizedReplayBuffer, MultiStepInnerModel, RLInnerModelCallBack
from RL.TorchModels.DQN import DQNInput, ConvDQNInput, DQNOutput, DQNCombine, NAFOutput

import torch.nn.functional as F

from RL.TorchRLBases import TorchRLInnerModelDecorator



class SyncTargetNetWorkStrategy(RLInnerModelCallBack,metaclass=SerializeMeta):
    '''
    适用于OnPolicy用于更新target network
    '''


class StepSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self,steps_interval=1000):
        self.steps_interval =steps_interval
        self.steps=0
    def onEpisodeFinished(self, episode,total_reward,model):
        pass
    def onStepFinished(self, episode, state, next_state, action, reward, done, info,model):
        self.steps=self.steps+1
        if self.steps % self.steps_interval == 0:
            model.target_net.load_state_dict(model.policy_net.state_dict())



class EpisodeSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self,episode_interval=5):
        self.episode_interval =episode_interval
    def onEpisodeFinished(self, episode,total_reward,model):
        if episode % self.episode_interval == 0:
            model.target_net.load_state_dict(model.policy_net.state_dict())
    def onStepFinished(self, episode, state, next_state, action, reward, done, info,model):
        pass

class SoftSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self,tau):
        self.tau=tau
    def onEpisodeFinished(self, episode,total_reward,model):
        pass
    def onStepFinished(self, episode, state, next_state, action, reward, done, info,model):
        for target_param, policy_param in zip(
                model.target_net.parameters(), model.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )


class HRLDQNInnerModel(HRLNodeInnerModelDecorator(TorchRLInnerModelDecorator(MultiStepInnerModel))):
    '''
    '''
    columns = ["state", "action", "reward", "next_state", "done", "length"]
    '''
    有问题待修改
    '''

    def __deserialize_post__(self):
        """反序列化后重新启动线程"""
        if not hasattr(self, 'memory'):
            self.set_default_memory()
    
    __exclude__ = ['memory']


    support_state_types=[gymnasium.spaces.box.Box,gym.spaces.box.Box]
    support_action_types=[gymnasium.spaces.Discrete,gymnasium.spaces.MultiDiscrete,gym.spaces.Discrete,gym.spaces.MultiDiscrete,int]
    def __init__(self, state_space, action_space, device=None, memory=None, n_steps=1, batch_size=32,
                 memory_capacity=10000, gamma=0.99, is_primitive=False, exploration_strategy=None,
                 sync_target_strategy:SyncTargetNetWorkStrategy=None):
        # 父类会调用set_default_memory,本类中该方法需要n_steps因此先设置
        self.n_steps = n_steps
        super().__init__(state_space=state_space,action_space=action_space,device=device, memory=memory, is_primitive=is_primitive, n_steps=n_steps,
                         batch_size=batch_size, gamma=gamma, memory_capacity=memory_capacity,
                         exploration_strategy=exploration_strategy)
        self.sync_target_strategy = sync_target_strategy if sync_target_strategy is not None else SoftSyncTargetNetWorkStrategy(tau=0.001)
        # 初始化双网络
        self.policy_net = self.get_policy_net()
        self.target_net = self.get_target_net()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.callbacks.append(self.sync_target_strategy)
    def get_policy_net(self):
        input_net=None
        if isinstance(self.action_space, int):
            output_net=DQNOutput(64,(self.action_space,))
        else:
            output_net=DQNOutput(64,self.action_space.shape)
        if isinstance(self.state_space, int):
            input_net =DQNInput(self.state_shape, 64)
        elif isinstance(self.state_space, (gymnasium.spaces.box.Box,gym.spaces.box.Box)):
            if len(self.state_space.shape) == 1:
                input_net =DQNInput(self.state_space.shape,64)

            elif len(self.state_space.shape) == 2 or len(self.state_space.shape) == 3:
                input_net=ConvDQNInput(self.state_space.shape, 64).to(self.device)
            else:
                raise ValueError("state_shape must be int or tuple/list with length 1、2、3")
        else:
            raise ValueError("state_shape must be int or tuple/list with length 1、2、3")

        return DQNCombine(input_net,output_net).to(self.device)
    def get_target_net(self):
        return self.get_policy_net()

    def set_default_memory(self):
        self.memory = PrioritizedReplayBuffer(capacity=self.memory_capacity, n_steps=self.n_steps, alpha=0.6, beta=0.4,
                                              columns=self.__class__.columns)

    def get_target_value(self, data, to_basic_type):
        #[batch_size,action_dim]
        if isinstance(data, dict):
            data1 = {column: [data[column]] for column in self.__class__.columns}
        else:
            data1 = {column: [d[column] for d in data] for column in self.__class__.columns}
        states = torch.stack(data1["state"])
        rewards = torch.stack(data1["reward"])
        next_states = torch.stack(data1["next_state"])
        dones = torch.stack(data1["done"])
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])

        actions = torch.stack(data1["action"])
        next_q = self.target_net(next_states).max(-1)[0].detach()
        tmp=((1 - dones) * self.gamma ** lengths).unsqueeze(1)
        if self.is_primitive:
            #rewards:[32,] next_q:[32,29] rewards变为[32,29]
            rewards=rewards.unsqueeze(1)
            expected_q = rewards + tmp*next_q
        else:
            expected_q = rewards + tmp*next_q

        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

    def get_current_value(self, data, to_basic_type):
        # [batch_size,action_dim]
        if isinstance(data, dict):
            data1 = {column: [data[column]] for column in self.__class__.columns}
        else:
            data1 = {column: [d[column] for d in data] for column in self.__class__.columns}

        states = torch.stack(data1["state"])
        rewards = torch.stack(data1["reward"])
        next_states = torch.stack(data1["next_state"])
        dones = torch.stack(data1["done"])
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])
        actions = torch.stack(data1["action"])
        flag=False
        '''
            data1=
        
        
        '''
        if len(actions.shape)==1:
            #action是一个数字的情况
            actions = actions.unsqueeze(1)
        '''
        actions:[batch_size,action_dim]
        action是一个值:[batch_size,] =>[batch_size,action_dim]
    
        '''
        actions = actions.unsqueeze(len(actions.shape))#[batch_size,action_dim,1]
        '''
        q_values=[batch_size,action_dim,action_options]
        [batch_size,action_options]=>[batch_size,1,action_options]
        
        '''
        q_values = self.policy_net(states)

        if len(q_values.shape)==2:
            #[batch_size,action_options]
            q_values = q_values.unsqueeze(1)
        elif len(q_values.shape)==3:
            #[batch_size,action_dim,action_options]
            pass

        current_q = q_values.gather(-1, actions).squeeze(-1)

        if to_basic_type:
            return current_q.detach().cpu().numpy()
        return current_q


    def _update(self, indices, batch_data, weights):
        '''
        current_q = []
        expected_q = []
        #太耗费时间了
        for data in batch_data:
            current = self.get_current_value(data,False)
            expected= self.get_target_value(data,False)
            current_q.append(current)
            expected_q.append(expected)
        # 保留梯度信息，维持反向传播
        current_q=torch.stack(current_q)
        expected_q=torch.stack(expected_q)
        '''
        start_time = time.time()

        current_q = self.get_current_value(batch_data, to_basic_type=False)
        expected_q = self.get_target_value(batch_data, to_basic_type=False)
        # 计算损失 输出一个值
        # loss = nn.MSELoss()(current_q, expected_q.unsqueeze(1))

        weights = torch.asarray(weights, device=self.device)
        weights=weights.unsqueeze(1)
        # reduction='none'表示不对batch平方差求平均
        loss = (weights * nn.MSELoss(reduction='none')(current_q, expected_q)).mean()
        start = time.time()
        self.memory.update(indices, current_q.detach().cpu().numpy(), expected_q.detach().cpu().numpy())
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("update_time:",time.time()-start_time,"s")

    def get_action_scores(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze()
        return q_values



class HRLNAFInnerModel(HRLDQNInnerModel):
    support_state_types = [gym.spaces.Box,gymnasium.spaces.Box,int]
    support_action_types = [gym.spaces.Box,gymnasium.spaces.Box]
    def _select_action(self,state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        _, mu, P,L = self.policy_net(state)

        #a_max、a_min没有设置需要修改
        a_max=1
        a_min=-1
        eps_min=0.01
        eps_start = 1
        T=200000
        decay_rate_step = (eps_min / eps_start) ** (1.0 / T)
        eps= max(eps_min, eps_start * (decay_rate_step ** self.steps_done))
        sigma = eps * (a_max - a_min)/2
        #action = mu + sigma * torch.randn_like(mu)
        if self.steps_done%100==0:
            print("sigma:",sigma)

        # add noise to action mu:
        #dist = MultivariateNormal(mu.squeeze(-1),scale_tril=L)
        '''
        P不用求逆，NAF中本身就是精度矩阵
        '''
        dist = MultivariateNormal(mu.squeeze(-1),precision_matrix= P)
        # dist = Normal(action_value.squeeze(-1), 1)
        action = sigma*dist.sample()

        action=torch.clamp(action,a_min,a_max)

        return action.detach().cpu().numpy()[0]

    def get_policy_net(self):
        input_net = None
        #判断action_space
        output_net = NAFOutput(64, self.action_space.shape[0])
        if isinstance(self.state_space,int):
            input_net =DQNInput(self.state_space,64)
        else:
            if len(self.state_space.shape) == 1:
                input_net = DQNInput(self.state_space.shape, 64)
            elif len(self.state_space.shape) == 2 or len(self.state_space.shape) == 3:
                input_net = ConvDQNInput(self.state_space.shape, 64).to(self.device)
            else:
                raise ValueError("state_shape must be Box with the length of shape： 1、2、3")

        return DQNCombine(input_net, output_net).to(self.device)
    def get_current_value(self, data, to_basic_type):
        # [batch_size,action_dim]
        if isinstance(data, dict):
            data1 = {column: [data[column]] for column in self.__class__.columns}
        else:
            data1 = {column: [d[column] for d in data] for column in self.__class__.columns}

        states = torch.stack(data1["state"])
        rewards = torch.stack(data1["reward"])
        next_states = torch.stack(data1["next_state"])
        dones = torch.stack(data1["done"])
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])
        actions = torch.stack(data1["action"])
        flag=False
        if len(actions.shape)==1:
            #action是一个数字的情况
            actions = actions.unsqueeze(1)
            flag=True
        actions = actions.unsqueeze(len(actions.shape))#[batch_size,action_dim,1]
        V,mu,P,L = self.policy_net(states)
        current_q=V
        if flag:
            #action如果是一个值则[batch_size,新增action_shape=1,]
            current_q =current_q.unsqueeze(1)

        if to_basic_type:
            return current_q.detach().cpu().numpy()
        return current_q

    def get_target_value(self, data, to_basic_type):
        # [batch_size,action_dim]
        if isinstance(data, dict):
            data1 = {column: [data[column]] for column in self.__class__.columns}
        else:
            data1 = {column: [d[column] for d in data] for column in self.__class__.columns}
        states = torch.stack(data1["state"])
        rewards = torch.stack(data1["reward"])
        next_states = torch.stack(data1["next_state"])
        dones = torch.stack(data1["done"])
        #n_steps维护时可能会修改
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])

        actions = torch.stack(data1["action"])
        V, mu, P,L= self.target_net(next_states)
        mu = mu.detach()
        V=V.detach()
        P=P.detach()
        diff = (actions - mu)
        # A: [B,1,D], B: [B,D,D]
        # 我们对 D 这个维度求和，留下 batch,1,K,D
        tmp = torch.einsum('bld,bdm->blm', diff.unsqueeze(1), P)#[32,1,29]
        #A:[B,l,M] B:[B,M,1]
        tmp1 = torch.einsum('blm,bma->bla',tmp,diff.unsqueeze(2))

        A = -0.5 *tmp1#[1,1,1]
        A=A.squeeze(-1).squeeze(-1)
        next_q = V + A
        tmp = ((1 - dones) * self.gamma ** lengths).unsqueeze(1)
        if self.is_primitive:
            rewards = rewards.unsqueeze(1)
            expected_q = rewards + tmp * next_q
        else:
            expected_q = rewards + tmp * next_q
        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q
    def save_best_model(self):
        self.best_model={"policy_net":self.policy_net.parameters(),
                         "target_net":self.target_net.parameters()}
    def load_best_model(self):
        self.policy_net=torch.load(self.best_model["policy_net"])
        self.target_net=torch.load(self.best_model["target_net"])
