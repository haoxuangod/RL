import random
import time

import gym
import gymnasium
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, MultiDiscrete
from torch import optim, nn
from torch.distributions import MultivariateNormal
from torch.optim.lr_scheduler import CosineAnnealingLR

from Decorator.Meta import SerializeMeta
from RL.HierarchicalRL.HRLBases import HRLNodeInnerModelDecorator
from RL.RLBases import PrioritizedReplayBuffer, RLInnerModelCallBack, OffPolicyInnerModel, ReplayBuffer, \
    MultiStepInnerModelDecorator, ContinuousExplorationStrategy, RLTimeSeriesInnerModelDecorator, \
    ExperienceReplayBuffer, OUNoise

from RL.RLModels.ValueOffPolicyInnerModel import ValueOffPolicyInnerModel
from RL.TorchModels.BaseNet import CombineNet, EnsembleNet
from RL.TorchModels.DQN import  DQNOutput, NAFOutput

import torch.nn.functional as F

from RL.TorchModels.ModelInput import MLPInput, CNNInput, LSTMInput
from RL.TorchRLBases import TorchRLInnerModelDecorator


class SyncTargetNetWorkStrategy(RLInnerModelCallBack, metaclass=SerializeMeta):
    '''
    适用于OnPolicy用于更新target network
    '''


class StepSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self, steps_interval=1000):
        self.steps_interval = steps_interval
        self.steps = 0

    def onEpisodeFinished(self, episode, total_reward, model,env):
        pass

    def onStepFinished(self, episode, state, next_state, action, reward, done, info, model,is_eval=False):
        if not is_eval:
            self.steps = self.steps + 1
            if self.steps % self.steps_interval == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())


class EpisodeSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self, episode_interval=5):
        self.episode_interval = episode_interval

    def onEpisodeFinished(self, episode, total_reward, model,env,is_eval=False):
        if is_eval:
            if episode % self.episode_interval == 0:
                model.target_net.load_state_dict(model.policy_net.state_dict())

    def onStepFinished(self, episode, state, next_state, action, reward, done, info, model,is_eval=False):
        pass


class SoftSyncTargetNetWorkStrategy(SyncTargetNetWorkStrategy):
    def __init__(self, tau):
        self.tau = tau

    def onEpisodeFinished(self, episode, total_reward, model,env,is_eval=False):
        pass

    def onUpdateFinished(self,update_cnt,model):
        for target_param, policy_param in zip(
                model.target_net.parameters(), model.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

        model.target_net.output_net[0].popart.mu = model.policy_net.output_net[0].popart.mu
        model.target_net.output_net[0].popart.sigma = model.policy_net.output_net[0].popart.sigma
        model.target_net.output_net[1].popart.mu = model.policy_net.output_net[1].popart.mu
        model.target_net.output_net[1].popart.sigma = model.policy_net.output_net[1].popart.sigma

    '''
    def onStepFinished(self, episode, state, next_state, action, reward, done, info, model,is_eval=False):
        for target_param, policy_param in zip(
                model.target_net.parameters(), model.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
        for target_param, policy_param in zip(
                model.target_net1.parameters(), model.policy_net.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )
    '''


class DQNInnerModel(TorchRLInnerModelDecorator(MultiStepInnerModelDecorator(ValueOffPolicyInnerModel))):
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

    support_state_types = [gymnasium.spaces.box.Box, gym.spaces.box.Box]
    support_action_types = [gymnasium.spaces.Discrete, gymnasium.spaces.MultiDiscrete, gym.spaces.Discrete,
                            gym.spaces.MultiDiscrete, int]

    def __init__(self, state_space, action_space, device=None, memory=None, n_steps=1, batch_size=32,
                 memory_capacity=10000, update_freq=4,lr=1e-4,gamma=0.99, steps_before_update=1,record_stats_dir=None,
                 exploration_strategy=None,state_processor=None,sync_target_strategy: SyncTargetNetWorkStrategy = None):
        # 父类会调用set_default_memory,本类中该方法需要n_steps因此先设置
        self.n_steps = n_steps
        super().__init__(state_space=state_space, action_space=action_space, device=device, memory=memory,
                         n_steps=n_steps,record_stats_dir=record_stats_dir,
                         batch_size=batch_size,lr=lr,gamma=gamma, memory_capacity=memory_capacity,update_freq=update_freq,
                         steps_before_update=steps_before_update,exploration_strategy=exploration_strategy,state_processor=state_processor)
        self.sync_target_strategy = sync_target_strategy if sync_target_strategy is not None else self.get_default_sync_target_strategy()
        # 初始化双网络
        self.policy_net = self._build_policy_net()
        self.target_net = self._build_target_net()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200*2000, eta_min=1e-6)
        self.callbacks.append(self.sync_target_strategy)

    def get_default_sync_target_strategy(self):
        return SoftSyncTargetNetWorkStrategy(tau=1e-4)
    def get_best_model(self):
        return {"policy_net": self.policy_net.state_dict(),
                           "target_net": self.target_net.state_dict()}
    def load_best_model(self):
        if not len(self.best_model)==0:
            self.policy_net.load_state_dict(self.best_model["policy_net"])
            self.target_net.load_state_dict(self.best_model["target_net"])
    def _build_policy_output_net(self):
        if isinstance(self.action_space, int):
            output_net = DQNOutput(64, (self.action_space,))
        else:
            output_net = DQNOutput(64, self.action_space.shape)
        return output_net
    def _build_policy_input_net(self):
        if isinstance(self.state_space, int):
            input_net = MLPInput(self.state_space, 64)
        elif isinstance(self.state_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            if len(self.state_space.shape) == 1:
                input_net = MLPInput(self.state_space.shape, 64)

            elif len(self.state_space.shape) == 2 or len(self.state_space.shape) == 3:
                input_net = CNNInput(self.state_space.shape, 64).to(self.device)
            else:
                raise ValueError("state_shape must be int or tuple/list with length 1、2、3")
        else:
            raise ValueError("state_shape must be int or tuple/list with length 1、2、3")
        return input_net
    def _build_policy_net(self):
        input_net = self._build_policy_input_net()
        output_net = self._build_policy_output_net()
        return CombineNet(input_net, output_net,log_dir=self.record_stats_dir).to(self.device)
    def _build_target_output_net(self):
        return self._build_policy_output_net()
    def _build_target_input_net(self):
        return self._build_policy_input_net()
    def _build_target_net(self):
        input_net = self._build_target_input_net()
        output_net = self._build_target_output_net()
        return CombineNet(input_net, output_net,log_dir=self.record_stats_dir).to(self.device)

    def set_default_memory(self):
        self.memory = PrioritizedReplayBuffer(capacity=self.memory_capacity, n_steps=self.n_steps, alpha=0.6, beta=0.4,
                                              columns=self.__class__.columns)

        self.memory=ExperienceReplayBuffer(capacity=self.memory_capacity,n_steps=self.n_steps,gamma=self.gamma,columns=self.__class__.columns)

    def get_max_action_value(self,states, to_basic_type):
        max_value = self.target_net(states).max(-1)[0]
        if to_basic_type:
            return max_value.detach().cpu().numpy()
        return max_value

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
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])

        actions = torch.stack(data1["action"])
        next_q = self.target_net(next_states).max(-1)[0].detach()

        tmp = ((1 - dones) * self.gamma ** lengths).unsqueeze(1)

        if self.is_primitive:
            # rewards:[32,] next_q:[32,29] rewards变为[32,29]
            rewards = rewards.unsqueeze(1)
            expected_q = rewards + tmp * next_q
        else:
            expected_q = rewards + tmp * next_q

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
        flag = False
        if len(actions.shape) == 1:
            # action是一个数字的情况
            actions = actions.unsqueeze(1)
        '''
        actions:[batch_size,action_dim]
        action是一个值:[batch_size,] =>[batch_size,action_dim]

        '''
        actions = actions.unsqueeze(len(actions.shape))  # [batch_size,action_dim,1]
        '''
        q_values=[batch_size,action_dim,action_options]
        [batch_size,action_options]=>[batch_size,1,action_options]

        '''
        q_values = self.policy_net(states)

        if len(q_values.shape) == 2:
            # [batch_size,action_options]
            q_values = q_values.unsqueeze(1)
        elif len(q_values.shape) == 3:
            # [batch_size,action_dim,action_options]
            pass
        current_q = q_values.gather(-1, actions).squeeze(-1)

        if to_basic_type:
            return current_q.detach().cpu().numpy()
        return current_q

    def predict(self, state):
        self.policy_net.eval()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze()
        self.policy_net.train()
        return q_values
    def _onEpisodeFinished(self, episode, total_reward,env):
        pass

       #self.scheduler.step()

class DoubleDQNInnerModel(DQNInnerModel):
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
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])

        actions = torch.stack(data1["action"])

        next_q_policy = self.policy_net(next_states)
        next_actions = next_q_policy.argmax(dim=-1, keepdim=True)
        next_q = self.target_net(next_states).gather(-1, next_actions).squeeze(-1)

        tmp = ((1 - dones) * self.gamma ** lengths).unsqueeze(1)
        if self.is_primitive:
            # rewards:[32,] next_q:[32,29] rewards变为[32,29]
            rewards = rewards.unsqueeze(1)
            expected_q = rewards + tmp * next_q
        else:
            expected_q = rewards + tmp * next_q

        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

class TimeSeriesDQNInnerModel(RLTimeSeriesInnerModelDecorator(DQNInnerModel)):
    def _build_policy_input_net(self):
        # def __init__(self, input_size, hidden_size=64, output_size=1,dropout_rate=0.5):
        return LSTMInput(input_size=self.window_size, hidden_size=128,output_size=64,dropout_rate=0.3)
class NAFExplorationStrategy(ContinuousExplorationStrategy):
    def __init__(self, a_max=1,a_min=-1,eps_start=1,eps_min=0.01,T=200000):
        self.a_max = a_max
        self.a_min = a_min
        self.eps_start = eps_start
        self.eps_min =eps_min
        self.T=T
        self.decay_rate_step = (eps_min / eps_start) ** (1.0 / T)
    def reset(self):
        pass
    def select_action(self, mean_action,*args,step=None,is_eval=False,episode=None,**kwargs):

        if is_eval:
            action= mean_action
            action = torch.clamp(action, self.a_min, self.a_max)
            return action.detach().cpu().numpy()

        eps = max(self.eps_min, self.eps_start * (self.decay_rate_step ** step))
        sigma = eps * (self.a_max - self.a_min) / 2
        # action = mu + sigma * torch.randn_like(mu)
        if step % 100 == 0:
            print("sigma:", sigma,"mu:",mean_action)

        #dist = MultivariateNormal(mean_action, scale_tril=args[2])

        '''
        Sigma_raw = torch.inverse(P1)
        
        '''
        P=args[2]
        Lc = torch.linalg.cholesky(P)  # Lc · Lcᵀ = P
        Sigma_raw = torch.cholesky_inverse(Lc)
        # 强制对称：
        Sigma = 0.5 * (Sigma_raw + Sigma_raw.transpose(-1, -2))  # Σ = P⁻¹

        dist = MultivariateNormal(mean_action, Sigma)
        # dist = Normal(action_value.squeeze(-1), 1)
        action = sigma * dist.sample()

        #action=dist.sample()
        action = torch.clamp(action, self.a_min, self.a_max)

        return action.detach().cpu().numpy()[0]
class NAFInnerModel(DQNInnerModel):
    support_state_types = [gym.spaces.Box, gymnasium.spaces.Box, int]
    support_action_types = [gym.spaces.Box, gymnasium.spaces.Box]
    def get_default_continuous_exploration_strategy(self):
        return NAFExplorationStrategy(a_max=self.action_space.high[0], a_min=self.action_space.low[0], eps_min=0.01,eps_start=1,T=1000000)


    def predict(self, state):
        self.policy_net.eval()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu,V,Q,L = self.policy_net(state)
        self.policy_net.train()
        return mu,V,Q,L


    '''
    def _select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mu,_,P,L = self.policy_net(state)

        # a_max、a_min没有设置需要修改
        a_max = 1
        a_min = -1
        eps_min = 0.01
        eps_start = 1
        T = 200000
        decay_rate_step = (eps_min / eps_start) ** (1.0 / T)
        eps = max(eps_min, eps_start * (decay_rate_step ** self.steps_done))
        sigma = eps * (a_max - a_min) / 2
        # action = mu + sigma * torch.randn_like(mu)
        if self.steps_done % 100 == 0:
            print("sigma:", sigma)

        # add noise to action mu:
        # dist = MultivariateNormal(mu.squeeze(-1),scale_tril=L)

        #P不用求逆，NAF中本身就是精度矩阵
        dist = MultivariateNormal(mu.squeeze(-1),scale_tril=L)
        # dist = Normal(action_value.squeeze(-1), 1)
        action = sigma * dist.sample()

        action = torch.clamp(action, a_min, a_max)

        return action.detach().cpu().numpy()[0]
    '''
    def _build_policy_output_net(self):
        output_net = NAFOutput(64, self.action_space.shape[0],
                               a_min=self.action_space.low,a_max=self.action_space.high)
        return output_net

    def get_max_action_value(self, states,to_basic_type=True):
        mu, V, _, _ = self.target_net(states)
        if to_basic_type:
            return V.detach().cpu().numpy()
        return V

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
        flag = False
        if len(actions.shape) == 1:
            # action是一个数字的情况
            actions = actions.unsqueeze(1)
            flag = True

        mu, V,Q,L = self.policy_net(states,actions)
        current_q=Q

        if flag:
            # action如果是一个值则[batch_size,新增action_shape=1,]
            current_q = current_q.unsqueeze(1)

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
        # n_steps维护时可能会修改
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])

        actions = torch.stack(data1["action"])
        mu,V,_,_ = self.target_net(next_states)
        next_q=V.detach()
        tmp = ((1 - dones) * self.gamma ** lengths).unsqueeze(1)
        expected_q = rewards + tmp * next_q

        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

class ConservativeDoubleNAFInnerModel(NAFInnerModel):
    def __init__(self,*args,critic_lr=1e-4,actor_lr=1e-4,actor_delay=3,**kwargs):
        super().__init__(*args,**kwargs)
        # —— 分离 Actor 参数 —— #
        self.actor_params = []
        self.actor_params+=self.policy_net.input_net.parameters()
        output_net=self.policy_net.output_net[0]
        output_net1=self.policy_net.output_net[1]
        for head in output_net.mu_heads:
            self.actor_params += list(head.parameters())
        self.ema_max=None
        self.ema_max1 = None
        self.ema_cql_loss=None

        self.alpha_min = 1e-4
        self.alpha_max = 10
        self.target_ratio = 0.1  # 想让 cons_loss ≈ 0.1 * td_loss
        # —— 分离 Critic 参数 —— #
        self.critic_params = []
        self.critic_params += self.policy_net.input_net.parameters()
        # 1) PopArt 里的所有参数
        self.critic_params += list(output_net.popart.parameters())
        # 2) L-entry 矩阵的参数
        self.critic_params += list(output_net.l_entry.parameters())
        self.critic_params += list(output_net.l_entry.parameters())

        # 1) PopArt 里的所有参数
        self.critic_params += list(output_net1.popart.parameters())
        # 2) L-entry 矩阵的参数
        self.critic_params += list(output_net1.l_entry.parameters())
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr
        self.actor_delay=actor_delay
        # 然后为它们准备两个 optimizer
        self.optimizer_critic = torch.optim.Adam(self.critic_params, lr=self.critic_lr)
        self.optimizer_actor = torch.optim.Adam(self.actor_params, lr=self.actor_lr)

        self.scheduler_critic= CosineAnnealingLR(self.optimizer_critic, T_max=200*2000, eta_min=1e-6)
        self.scheduler_actor= CosineAnnealingLR(self.optimizer_actor, T_max=200*2000,eta_min=1e-6)
    def predict(self, state):
        self.policy_net.eval()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu,V,Q,L = self.policy_net(state,index=0)
        self.policy_net.train()
        return mu,V,Q,L

    def _build_policy_output_net(self):
        output_net1 = NAFOutput(64, self.action_space.shape[0],
                               a_min=self.action_space.low, a_max=self.action_space.high)
        output_net2 = NAFOutput(64, self.action_space.shape[0],
                                a_min=self.action_space.low, a_max=self.action_space.high)
        output_net=EnsembleNet(output_net1,output_net2)
        return output_net
    def get_best_model(self):
        return {"policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict()}
    def load_best_model(self):
        if not len(self.best_model)==0:
            self.policy_net.load_state_dict(self.best_model["policy_net"])
            self.target_net.load_state_dict(self.best_model["target_net"])
    def get_default_sync_target_strategy(self):
        #3e-4
        return SoftSyncTargetNetWorkStrategy(tau=1e-4)
        #return StepSyncTargetNetWorkStrategy(steps_interval=1000)
    def get_cql_loss(self, data,sample_num=20):
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
        flag = False
        if len(actions.shape) == 1:
            # action是一个数字的情况
            actions = actions.unsqueeze(1)
            flag = True
        mu, V, Q, P = self.policy_net(states, actions)
        #assert V.mean()>1000
        Lc = torch.linalg.cholesky(P)  # Lc · Lcᵀ = P
        Sigma_raw = torch.cholesky_inverse(Lc)  # Σ = P⁻¹
        #Sigma_raw = torch.inverse(P)
        # 强制对称：
        Sigma = 0.5 * (Sigma_raw + Sigma_raw.transpose(-1, -2))  # Σ = P⁻¹
        dist = MultivariateNormal(mu.squeeze(-1),covariance_matrix=Sigma)
        #[sample_num,batch_size,action_dim]
        random_actions=dist.sample((sample_num,))
        low = torch.tensor(self.action_space.low, device=random_actions.device)  # [action_dim]
        high = torch.tensor(self.action_space.high, device=random_actions.device)

        # random_actions: [sample_num, batch_size, action_dim]
        random_actions = torch.max(torch.min(random_actions, high), low)
        batch_size=states.shape[0]
        state_shape=states.shape[1:]
        action_shape=actions.shape[1:]
        # 展平方便并行计算 Q
        flat_states = states.unsqueeze(0).repeat(sample_num,1,1).view(-1, *state_shape)
        flat_rand_actions = random_actions.view(-1, *action_shape).detach()
        mu1, V1, Q1, P1 = self.policy_net(flat_states, flat_rand_actions)
        #assert (torch.abs(V1-Q1) > 500).any()
        Q1=Q1.view(sample_num,batch_size,-1)
        Q1=torch.logsumexp(Q1, dim=0)
        #Q1=Q1.mean(dim=0)
        Q2=(Q1-Q)
        Q2_clamped = torch.clamp(Q2, min=-20.0, max=20.0)
        cur_max=Q2_clamped.abs().max().item()
        if self.ema_max1 is None:
            self.ema_max1 = cur_max
        else:
            # EMA 估计 max |Q|
            self.ema_max1 = 0.99 * self.ema_max1 + 0.01 * cur_max

        scaler = 1.0 / (self.ema_max1 + 1e-6)
        q_norm = torch.tanh(Q2_clamped * scaler)
        cql_loss = q_norm.mean()

        return cql_loss


    def _update(self, indices, batch_data, weights):

        critic_loss,current_q,expected_q=self.get_critic_loss(batch_data,weights,return_only_loss=False)
        self.memory.update(indices, current_q.detach().cpu().numpy(), expected_q.detach().cpu().numpy())

        #cql_loss=self.get_cql_loss(batch_data)
        '''
        if self.ema_cql_loss is None:
            self.ema_cql_loss=cql_loss
        else:
            self.ema_cql_loss=0.99*self.ema_cql_loss+0.01*cql_loss
        alpha = torch.abs((self.target_ratio * critic_loss) / (self.ema_cql_loss + 1e-6))
        alpha = float(max(self.alpha_min, min(self.alpha_max, alpha)))
        self.alpha = alpha
        cql_loss=0
        '''
        #self.alpha=2
        #loss=critic_loss+self.alpha*cql_loss
        loss=critic_loss

        # 反向传播
        self.optimizer_critic.zero_grad()

        for name, param in self.policy_net.named_parameters():
            if param.grad is not None:
                # 这里可以打印梯度的范数，或者直接打印梯度张量
                grad_norm = param.grad.norm().item()
                print(f"critic参数 `{name}` 收到梯度，梯度范数 = {grad_norm:.6f}")
            else:
                print(f"critic参数 `{name}` 没有收到梯度")
        loss.backward()
        #效果反而会差，在第400次更新后会突然梯度爆炸
        #torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)

        # —— 2. 延迟 Actor 更新 —— #
        if self.update_cnt % self.actor_delay == 0:
            if isinstance(batch_data, dict):
                data1 = [batch_data["state"]]
            else:
                data1 = [d["state"] for d in batch_data]
            states = torch.stack(data1)

            # Actor 目标：最大化 Q(s,μ(s)) → 最小化 −Q
            mu,V,Q,_ = self.policy_net(states,index=0)
            cur_max = min(V.detach().abs().max().item(), 20)
            if self.ema_max is None:
                self.ema_max = cur_max
            else:
                # EMA 估计 max |Q|
                self.ema_max = 0.99 * self.ema_max + 0.01 * cur_max
            mu1, V1, Q1, _ = self.target_net(states,index=0)
            mu1=mu1.detach()
            l2_reg = F.mse_loss(mu, mu1)
            lambda1=0.01

            scaler = 1.0 / (self.ema_max + 1e-6)
            q_norm = torch.tanh(-V.mean() * scaler)
            actor_loss = q_norm+lambda1*l2_reg
            #actor_loss =q_norm
            actor_loss =-V.mean()
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            # 遍历 model 的所有参数
            for name, param in self.policy_net.named_parameters():
                if param.grad is not None:
                    # 这里可以打印梯度的范数，或者直接打印梯度张量
                    grad_norm = param.grad.norm().item()
                    print(f"actor参数 `{name}` 收到梯度，梯度范数 = {grad_norm:.6f}")
                else:
                    print(f"actor参数 `{name}` 没有收到梯度")
            torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)
        if self.record_stats_dir is not None:
            self.summary_writer.add_scalar("loss/critic", critic_loss.detach().cpu().numpy().item(), self.update_cnt)
            #self.summary_writer.add_scalar("loss/ema_max/value", self.ema_max,
                                  #         self.update_cnt)
            if self.update_cnt % self.actor_delay == 0:
                self.summary_writer.add_scalar("loss/actor/value", actor_loss.detach().cpu().numpy().item(), self.update_cnt)

                self.summary_writer.add_scalar("loss/actor/l2_reg", l2_reg.detach().cpu().numpy().item(),
                                            self.update_cnt)
            #self.summary_writer.add_scalar("loss/cql/value", cql_loss.detach().cpu().numpy().item(), self.update_cnt)
            #self.summary_writer.add_scalar("loss/cql/alpha", self.alpha, self.update_cnt)
            self.summary_writer.add_scalar("loss/total", loss.detach().cpu().numpy().item(), self.update_cnt)

            # self.writer.add_scalar("current_q",current_q.detach().cpu().numpy().item(), self.update_cnt)
            current = current_q.detach().cpu().numpy().mean().item()
            target = expected_q.detach().cpu().numpy().mean().item()
            td_abs = abs(current - target)
            self.summary_writer.add_scalar("Q/current_mean", current, self.update_cnt)
            self.summary_writer.add_scalar("Q/target_mean", target, self.update_cnt)
            self.summary_writer.add_scalar("TD/abs_mean", td_abs, self.update_cnt)
            self.summary_writer.add_scalar("lr/scheduler_critic", self.optimizer_critic.param_groups[0]['lr'],
                                           self.update_cnt)
            self.summary_writer.add_scalar("lr/scheduler_actor", self.optimizer_actor.param_groups[0]['lr'],
                                           self.update_cnt)

            self.summary_writer.add_scalars("Q/current_and_target", {"current": current, "target": target},
                                            self.update_cnt)
        self.optimizer_critic.step()
        self.scheduler_critic.step()
        self.optimizer_actor.step()
        self.scheduler_actor.step()
    def get_current_value(self, data, to_basic_type,net=0):
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
        flag = False
        if len(actions.shape) == 1:
            # action是一个数字的情况
            actions = actions.unsqueeze(1)
            flag = True
        mu, V,Q,P = self.policy_net(states,actions,index=net)
        current_q=Q

        if flag:
            # action如果是一个值则[batch_size,新增action_shape=1,]
            current_q = current_q.unsqueeze(1)

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
        # n_steps维护时可能会修改
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])

        actions = torch.stack(data1["action"])
        '''
        mu2, V2, _, _=self.policy_net(next_states)
        '''
        mu,V, Q,_ = self.target_net(next_states,index=0)
        '''
        σ_noise=0.05
        noise_clip=0.1
        noise = (torch.randn_like(mu) * σ_noise).clamp(-noise_clip, noise_clip)
        mu_noisy = (mu + noise).clamp(-1, 1)
        '''
        mu_noisy = mu
        mu, V, Q, _ = self.target_net(next_states,mu_noisy,index=0)
        mu1, V1,Q1,_ = self.target_net(next_states,mu_noisy,index=1)
        next_q=torch.min(Q.detach(),Q1.detach())

        tmp = ((1 - dones) * self.gamma ** lengths).unsqueeze(1)
        expected_q = rewards + tmp * next_q
        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

    def get_critic_loss(self, data, weights, update_stats=True,return_only_loss=True):
        current_q = self.get_current_value(data, to_basic_type=False,net=0)
        current_q1=self.get_current_value(data, to_basic_type=False,net=1)
        expected_q = self.get_target_value(data, to_basic_type=False)

        popart = self.policy_net.output_net[0].popart
        if update_stats:
            # 1) 更新统计量并重标定网络输出层（不跟踪梯度）
            with torch.no_grad():
                popart.update_stats(expected_q)

        # 2) 用更新后的 μ、σ 做归一化
        mu, sigma = popart.mu.detach(), popart.sigma.detach()
        self.summary_writer.add_scalar("popart/mu", mu.mean(), self.update_cnt)
        self.summary_writer.add_scalar("popart/sigma", sigma.mean(), self.update_cnt)
        u_norm = (current_q - mu) / sigma
        y_norm = (expected_q - mu) / sigma

        clamp_val = 10.0  # 或 20.0，看你希望最大容许 10 个标准差以内
        u_norm = u_norm.clamp(min=-clamp_val, max=clamp_val)
        y_norm = y_norm.clamp(min=-clamp_val, max=clamp_val)

        popart1 = self.policy_net.output_net[1].popart
        if update_stats:
            # 1) 更新统计量并重标定网络输出层（不跟踪梯度）
            with torch.no_grad():
                popart1.update_stats(expected_q)

        # 2) 用更新后的 μ、σ 做归一化
        mu1, sigma1 = popart1.mu.detach(), popart1.sigma.detach()
        self.summary_writer.add_scalar("popart/mu1", mu1.mean(), self.update_cnt)
        self.summary_writer.add_scalar("popart/sigma1", sigma1.mean(), self.update_cnt)
        u_norm1 = (current_q1 - mu1) / sigma1

        clamp_val = 10.0
        u_norm1 = u_norm1.clamp(min=-clamp_val, max=clamp_val)
        # 3) 计算加权 SmoothL1 loss
        weights = torch.as_tensor(weights, device=self.device).unsqueeze(1)
        td_loss1 = (weights * nn.SmoothL1Loss(reduction='none')(u_norm, y_norm)).mean()
        td_loss2 = (weights * nn.SmoothL1Loss(reduction='none')(u_norm1, y_norm)).mean()
        td_loss = (td_loss1+td_loss2)/2
        return td_loss if return_only_loss else (td_loss, current_q, expected_q)
class DoubleNAFInnerModel(NAFInnerModel):
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
        # n_steps维护时可能会修改
        if isinstance(data1["length"][0], int):
            lengths = torch.tensor(data1["length"], dtype=torch.long, device=self.device)
        else:
            lengths = torch.stack(data1["length"])

        actions = torch.stack(data1["action"])
        mu1,V1,_,_=self.policy_net(next_states)
        mu, V, Q,_ = self.target_net(next_states,mu1)
        next_q=Q.detach()
        tmp = ((1 - dones) * self.gamma ** lengths).unsqueeze(1)
        expected_q = rewards + tmp * next_q
        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

class TimeSeriesNAFInnerModel(RLTimeSeriesInnerModelDecorator(NAFInnerModel)):
    def _build_policy_input_net(self):
        if isinstance(self.state_space,int):
            # def __init__(self, input_size, hidden_size=64, output_size=1,dropout_rate=0.5):
            return LSTMInput(input_size=self.state_space, hidden_size=128,output_size=64,dropout_rate=0.3)
        elif isinstance(self.state_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            if len(self.state_space.shape) == 1:
                return LSTMInput(input_size=self.state_space.shape[0], hidden_size=128,output_size=64,dropout_rate=0.3)