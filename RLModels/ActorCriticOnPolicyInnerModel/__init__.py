import gym
import gymnasium
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from RL.RLBases import OnPolicyInnerModel, StateProcessor
from RL.TorchModels.ActorCritic import ActorOutput, CriticOutput
from RL.TorchModels.BaseNet import CombineNet
from RL.TorchModels.ModelInput import MLPInput, CNNInput
from RL.TorchRLBases import TorchRLInnerModelDecorator


class ActorCriticOnPolicyInnerModel(TorchRLInnerModelDecorator(OnPolicyInnerModel)):
    support_state_types = [gym.spaces.Box, gymnasium.spaces.Box, int]
    support_action_types = [gym.spaces.Box, gymnasium.spaces.Box]
    def __init__(self,state_space,action_space,device=None,gamma=0.99,ent_coef=0.01,t_steps=5,lr_actor=3e-4,
        lr_critic=1e-3,state_processor:StateProcessor=None,exploration_strategy=None):
        super().__init__(device=device,state_space=state_space,action_space=action_space,state_processor=state_processor,exploration_strategy=exploration_strategy)
        self.lr_actor = lr_actor
        self.lr_critic =lr_critic
        #每收集t_steps的数据后更新一次
        self.t_steps = t_steps
        #当前收集了多少数据
        self._current_t=0
        self.buffer=[]
        self.gamma=gamma
        self.ent_coef=ent_coef
        self.actor_net = self._build_actor_net()
        self.critic_net = self._build_critic_net()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(),lr=lr_critic)
        self.actor_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=200, eta_min=1e-6)
        self.critic_scheduler =CosineAnnealingLR(self.critic_optimizer, T_max=200, eta_min=1e-6)
    def _onEpisodeFinished(self, episode, total_reward):
       self.actor_scheduler.step()
       self.critic_scheduler.step()
    def _select_action(self,state,is_eval=False):
        state = torch.FloatTensor(state).to(self.device)
        mu, std = self.actor_net(state)
        dist = torch.distributions.Normal(mu, std)
        actions = dist.rsample()
        return actions.detach().cpu().numpy()
    def _build_actor_input_net(self):
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
    def _build_actor_output_net(self):
        if isinstance(self.action_space, int):
            output_net = ActorOutput(64, self.action_space,hidden_size=128)
        else:
            #没有修改a_min和a_max
            output_net = ActorOutput(64, self.action_space.shape[0],hidden_size=128,
                                     a_min=self.action_space.low,a_max=self.action_space.high)
        return output_net
    def _build_actor_net(self):
        input_net = self._build_actor_input_net()
        output_net = self._build_actor_output_net()
        return CombineNet(input_net,output_net).to(self.device)
    def _build_critic_input_net(self):
        return self._build_actor_input_net()
    def _build_critic_output_net(self):
        output_net = CriticOutput(64, hidden_size=128)
        return output_net
    def _build_critic_net(self):
        input_net = self._build_critic_input_net()
        output_net = self._build_critic_output_net()
        return CombineNet(input_net, output_net).to(self.device)
    def _update(self, data):
        # 1. 收集数据、累加步数
        self._current_t += 1
        data1 = {key: self.convert_data(key, val) for key, val in data.items()}
        self.buffer.append(data1)
        if self._current_t < self.t_steps:
            return

        # 2. 将 buffer 中的数据按列提取
        #    假设 self.__class__.columns = ["state", "action", "reward", "next_state", "done", "length"]
        data1 = {col: [d[col] for d in self.buffer] for col in self.__class__.columns}

        # 3. 转为张量
        states = torch.stack(data1["state"])  # [T, N, state_dim]
        actions = torch.stack(data1["action"])  # [T, N, action_dim]
        rewards = torch.stack(data1["reward"])  # [T, N]
        next_states = torch.stack(data1["next_state"])  # [T, N, state_dim]
        dones = torch.stack(data1["done"]).float()  # [T, N]

        # 4. Bootstrap：计算最后一步的 value
        with torch.no_grad():
            last_value = self.critic_net(next_states[-1])  # [N]

        # 5. 计算多步回报和优势
        returns = torch.zeros_like(rewards)  # [T, N]
        advantages = torch.zeros_like(rewards)  # [T, N]
        R = last_value
        for t in reversed(range(self.t_steps)):
            R = rewards[t] + self.gamma * R * (1.0 - dones[t])
            returns[t] = R
            # 当前状态的值估计
            V = self.critic_net(states[t])
            advantages[t] = R - V
        #[T,N,action_dim]转换为[T*N,action_dim] N代表并行环境数量
        states_flat = states.view(-1, states.size(-1))
        actions_flat = actions.view(-1, actions.size(-1))
        returns_flat = returns.view(-1)
        advantages_flat = advantages.view(-1)

        # 7. 计算策略分布、log_prob 和熵
        mu, std = self.actor_net(states_flat)  # [T*N, action_dim] × 2
        dist = torch.distributions.Normal(mu, std)
        logp = dist.log_prob(actions_flat).sum(dim=-1)  # [T*N] 联合分布概率对数需要相加
        #熵衡量了分布的不确定性或随机性——熵越大，分布越“平坦”、越不可预测。
        entropy = dist.entropy().sum(dim=-1).mean()  # 标量，先得到联合分布然后对样本求平均

        # 8. Actor 和 Critic 的 Loss
        loss_actor = -(logp * advantages_flat.detach()).mean() \
                     - self.ent_coef * entropy
        values_pred = self.critic_net(states_flat)  # [T*N]
        loss_critic = (returns_flat - values_pred).pow(2).mean()

        # 9. 优化更新
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # 10. 重置计数与缓存
        self._current_t = 0
        self.buffer.clear()
