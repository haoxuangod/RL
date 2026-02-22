import os.path
import gymnasium
import torch
import numpy as np
from torch import optim, nn
from torch.distributions import MultivariateNormal, Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from RL.RLBases import RLInnerModelCallBack, MultiStepInnerModelDecorator, RLTimeSeriesInnerModelDecorator, \
    ExperienceReplayBuffer

from RL.RLModels.ValueOffPolicyInnerModel import ValueOffPolicyInnerModel
from RL.TorchModels.BaseNet import CombineNet, EnsembleNet
from RL.TorchModels.DQN import DQNOutput, NAFOutput, NAFPopArtOutput, MAFOutput, BNAFOutput

from RL.TorchModels.ModelInput import MLPInput, CNNInput, LSTMInput
from RL.TorchRLBases import TorchRLInnerModelDecorator
from RL.common.exploration.strategies.continuous import ContinuousExplorationStrategy
from RL.common.replaybuffer.PrioritizedBuffer import PrioritizedReplayBuffer
from RL.common.syncTargetNetWorkStrategies import SyncTargetNetWorkStrategy
from RL.common.syncTargetNetWorkStrategies.SoftSyncTargetNetWorkStrategy import SoftSyncTargetNetWorkStrategy
from RL.common.utils.decorator.Meta import SerializeMeta

class DQNInnerModel(TorchRLInnerModelDecorator(MultiStepInnerModelDecorator(ValueOffPolicyInnerModel))):

    columns = ["state", "action", "reward", "next_state", "done", "length"]
    columns_type_dict = {"state": np.float32, "action": np.int64, "reward": np.float32,
                         "next_state": np.float32, "done": np.float32, "length": np.int32}
    '''
    有问题待修改
    '''
    def __deserialize_post__(self):
        """反序列化后重新启动线程"""
        if not hasattr(self, 'memory'):
            self.set_default_memory()

    __exclude__ = ['memory']

    support_state_types = [gymnasium.spaces.box.Box]
    support_action_types = [gymnasium.spaces.Discrete, gymnasium.spaces.MultiDiscrete, int]

    def __init__(self, state_space, action_space, vec_env_num=None,device=None, memory=None, n_steps=1, batch_size=32,
                 memory_capacity=10000, update_freq=4,lr=1e-4,gamma=0.99, steps_before_update=1,tensorboard_log_dir=None,
                 exploration_strategy=None,state_processor=None,sync_target_strategy: SyncTargetNetWorkStrategy = None):
        # 父类会调用set_default_memory,本类中该方法需要n_steps因此先设置
        self.n_steps = n_steps
        super().__init__(state_space=state_space, action_space=action_space,vec_env_num=vec_env_num, device=device, memory=memory,
                         n_steps=n_steps,tensorboard_log_dir=tensorboard_log_dir,
                         batch_size=batch_size,lr=lr,gamma=gamma, memory_capacity=memory_capacity,update_freq=update_freq,
                         steps_before_update=steps_before_update,exploration_strategy=exploration_strategy,state_processor=state_processor)
        self.sync_target_strategy = sync_target_strategy if sync_target_strategy is not None else self.get_default_sync_target_strategy()

        # 初始化双网络
        self.policy_net = self._build_policy_net()
        self.target_net = self._build_target_net()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        #self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200*2000, eta_min=1e-6)
        self.callbacks.append(self.sync_target_strategy)
    def _set_action_shape(self):
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            self.action_shape = (1,)
        else:
            self.action_shape = self.action_space.shape
    def _setup_tensorboard(self, tensorboard_log_dir=None, base_tag=None):
        super()._setup_tensorboard(tensorboard_log_dir,base_tag)
        if self.sync_target_strategy.tensorboard_log_dir is None:
            tensorboard_log_dir = os.path.join(self.tensorboard_log_dir, self.sync_target_strategy.__class__.__name__)
            base_tag = self.base_tag + "/" + self.sync_target_strategy.__class__.__name__
            self.sync_target_strategy.setup_tensorboard(tensorboard_log_dir, base_tag)
        if self.policy_net.tensorboard_log_dir is None:
            tensorboard_log_dir = os.path.join(self.tensorboard_log_dir, "policy_net")
            base_tag = self.base_tag + "/policy_net"
            self.policy_net.setup_tensorboard(tensorboard_log_dir, base_tag)
        if self.target_net.tensorboard_log_dir is None:
            tensorboard_log_dir = os.path.join(self.tensorboard_log_dir, "target_net")
            base_tag = self.base_tag + "/target_net"
            self.target_net.setup_tensorboard(tensorboard_log_dir, base_tag)

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
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            output_net = DQNOutput(64, (int(self.action_space.n),))
        else:
            output_net = DQNOutput(64, self.action_space.shape)
        return output_net
    def _build_policy_input_net(self):
        if isinstance(self.state_space, int):
            input_net = MLPInput(self.state_space, 64)
        elif isinstance(self.state_space, (gymnasium.spaces.box.Box,)):
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
        return CombineNet(input_net, output_net,tensorboard_log_dir=self.tensorboard_log_dir).to(self.device)
    def _build_target_output_net(self):
        return self._build_policy_output_net()
    def _build_target_input_net(self):
        return self._build_policy_input_net()
    def _build_target_net(self):
        input_net = self._build_target_input_net()
        output_net = self._build_target_output_net()
        return CombineNet(input_net, output_net,tensorboard_log_dir=self.tensorboard_log_dir).to(self.device)

    def set_default_memory(self):
        '''
        self.memory = PrioritizedReplayBuffer(capacity=self.memory_capacity, n_steps=self.n_steps, alpha=0.6, beta=0.4,
                                              columns=self.__class__.columns)
        '''
        info_dict=self.columns_type_dict.copy()
        if self.vec_env_num is None:
            dic={"state" : self.state_space.shape,
                 "action" : self.action_shape,
                 "next_state" : self.state_space.shape,
                 "reward" : (1,),
                 "done" : (1,),
                 "length" : (1,)}
        else:
            dic = {"state": (self.vec_env_num,)+self.state_space.shape,
                   "action": (self.vec_env_num,)+self.action_shape,
                   "next_state": (self.vec_env_num,)+self.state_space.shape,
                   "reward": (self.vec_env_num,1),
                   "done": (self.vec_env_num,1),
                   "length": (self.vec_env_num,1)}
        info_dict={key:(val,dic[key]) for key,val in info_dict.items()}
        self.memory=ExperienceReplayBuffer(capacity=self.memory_capacity,n_steps=self.n_steps,gamma=self.gamma,
                                           columns=self.__class__.columns,columns_info_dict=info_dict)

    def get_max_action_value(self,states, to_basic_type):
        max_value = self.target_net(states).max(-1)[0]
        if to_basic_type:
            return max_value.detach().cpu().numpy()
        return max_value

    def get_target_value(self, next_states, rewards, dones, lengths, to_basic_type):
        # [batch_size,action_dim]
        next_q = self.target_net(next_states).max(-1)[0].detach()
        tmp = (1 - dones) * self.gamma ** lengths

        # 小心 rewards:[32,] next_q:[32,29] rewards变为[32,29] action会不会为[32,]
        expected_q = rewards + tmp * next_q

        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

    def get_current_value(self, states, actions, to_basic_type):
        actions = actions.long()
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
    def get_target_value(self, next_states, rewards, dones, lengths, to_basic_type):
        # [batch_size,action_dim]
        next_q_policy = self.policy_net(next_states)
        next_actions = next_q_policy.argmax(dim=-1, keepdim=True)
        next_q = self.target_net(next_states).gather(-1, next_actions).squeeze(-1)

        tmp = (1 - dones) * self.gamma ** lengths
        # 小心 rewards:[32,] next_q:[32,29] rewards变为[32,29] action会不会为[32,]
        expected_q = rewards + tmp * next_q

        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

class TimeSeriesDQNInnerModel(RLTimeSeriesInnerModelDecorator(DQNInnerModel)):
    def _build_policy_input_net(self):
        # def __init__(self, input_size, hidden_size=64, output_size=1,dropout_rate=0.5):
        return LSTMInput(input_size=self.window_size, hidden_size=128,output_size=64,dropout_rate=0.3)

def clamp_covariance(P: torch.Tensor,
                     lambda_min: float = 1e-3,
                     lambda_max: float = 1e3) -> torch.Tensor:
    """
    对称矩阵 P 的特征值截断，并计算其逆（即协方差 Σ）。

    输入:
      P: tensor, shape [..., n, n], 对称正定
      lambda_min: float, 特征值下限
      lambda_max: float, 特征值上限

    返回:
      Sigma: tensor, shape [..., n, n], 截断后 P^{-1}
    """
    # 1) 特征分解: P = U @ diag(e) @ U^T
    # e: [..., n], U: [..., n, n]
    e, U = torch.linalg.eigh(P)

    # 2) 截断特征值
    e_clamped = torch.clamp(e, min=lambda_min, max=lambda_max)  # [..., n]

    # 3) 构造倒数的对角矩阵 diag(1/e_clamped)
    inv_e = 1.0 / e_clamped                                  # [..., n]
    inv_diag = torch.diag_embed(inv_e)                       # [..., n, n]

    # 4) 重构 Σ = U @ inv_diag @ U^T
    Sigma = U @ inv_diag @ U.transpose(-2, -1)

    # （可选）如果你还想要“正则化后”的 P 矩阵：
    # P_clamped = U @ torch.diag_embed(e_clamped) @ U.transpose(-2, -1)
    # return Sigma, P_clamped

    return Sigma
class NAFExplorationStrategy(ContinuousExplorationStrategy):
    __exclude__ = [torch.utils.tensorboard.writer.SummaryWriter]
    def __deserialize_post__(self):
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
    def __init__(self, a_max,a_min,eps_start=1,eps_min=0.01,T=200000,tensorboard_log_dir=None):
        self.a_max = a_max
        self.a_min = a_min
        self.eps_start = eps_start
        self.eps_min =eps_min
        self.T=T
        self.decay_rate_step = (eps_min / eps_start) ** (1.0 / T)
        self.tensorboard_log_dir = tensorboard_log_dir

        if self.tensorboard_log_dir is not None:
            # 创建一个 writer，指定日志保存目录
            self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        else:
            self.summary_writer = None
        self.cnt=0
    def reset(self):
        pass
    def select_action(self, mean_action,*args,step=None,is_eval=False,episode=None,**kwargs):
        a_max=torch.from_numpy(self.a_max).to(mean_action.device)
        a_min=torch.from_numpy(self.a_min).to(mean_action.device)
        if is_eval:
            action= mean_action
            action = torch.max(torch.min(action, a_max), a_min)
            return action.detach().cpu().numpy()

        eps = max(self.eps_min, self.eps_start * (self.decay_rate_step ** step))
        sigma = eps * (a_max - a_min) / 2
        # action = mu + sigma * torch.randn_like(mu)
        if step % 100 == 0:
            print("sigma:", sigma,"mu:",mean_action)

        #dist = MultivariateNormal(mean_action, scale_tril=args[2])

        '''
        Sigma_raw = torch.inverse(P1)
        '''

        P=args[2]
        '''
        lambda_min=1
        lambda_max=100
        P=clamp_covariance(P,lambda_min,lambda_max)
        '''
        alpha=10
        P1=P*alpha
        Lc = torch.linalg.cholesky(P1)  # Lc · Lcᵀ = P
        eigenvalues, _ = torch.linalg.eigh(P1)#[B,action_dim]  # 返回升序排列的特征值
        lambda_min = eigenvalues[0,0]  # shape [B] 或 标量
        lambda_mean=eigenvalues.mean()
        Sigma_raw = torch.cholesky_inverse(Lc)
        # 强制对称：
        Sigma = 0.5 * (Sigma_raw + Sigma_raw.transpose(-1, -2))*sigma  # Σ = P⁻¹

        dist = MultivariateNormal(mean_action, Sigma)
        # dist = Normal(action_value.squeeze(-1), 1)
        action1 = dist.sample()
        action2=mean_action+sigma*0.01*torch.randn_like(mean_action)
        #action=(1-eps)*action1+eps*action2
        action=action2.unsqueeze(0)
        if self.tensorboard_log_dir is not None:
            with torch.no_grad():
                delta=torch.abs(action-mean_action)
                self.summary_writer.add_scalar('NAFExploration/sigma/max', Sigma.max(), self.cnt)
                self.summary_writer.add_scalar('NAFExploration/lambda/min', lambda_min, self.cnt)
                self.summary_writer.add_scalar('NAFExploration/lambda/mean', lambda_min, self.cnt)
                self.summary_writer.add_scalar('NAFExploration/delta/mean',delta.mean() , self.cnt)
                self.summary_writer.add_scalar('NAFExploration/delta/max', delta.max(), self.cnt)
        #action=dist.sample()

        #action=mean_action+0.2*sigma*torch.randn_like(mean_action)
        action = torch.max(torch.min(action, a_max), a_min)
        self.cnt=self.cnt+1
        return action.detach().cpu().numpy()[0]
class NAFInnerModel(DQNInnerModel):
    support_state_types = [gymnasium.spaces.Box, int]
    support_action_types = [gymnasium.spaces.Box]
    columns_type_dict = {"state": np.float32, "action": np.float32, "reward": np.float32,
                         "next_state": np.float32, "done": np.float32, "length": np.int32}
    def get_default_continuous_exploration_strategy(self):
        if self.tensorboard_log_dir==None:
            tensorboard_log_dir=None
        else:
            tensorboard_log_dir=os.path.join(self.tensorboard_log_dir,"NAFExploration")
        return NAFExplorationStrategy(a_max=self.action_space.high, a_min=self.action_space.low, eps_min=0.01,
                                      eps_start=1,T=1000000,tensorboard_log_dir=tensorboard_log_dir)


    def predict(self, state):
        self.policy_net.eval()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu,V,Q,P = self.policy_net(state)
        self.policy_net.train()
        return mu,V,Q,P


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

    def get_current_value(self, states,actions, to_basic_type):
        mu, V, Q, P = self.policy_net(states,actions)
        current_q=Q.unsqueeze(-1)

        if to_basic_type:
            return current_q.detach().cpu().numpy()
        return current_q

    def get_target_value(self, next_states,rewards,dones,lengths, to_basic_type):

        mu,V,_,_ = self.target_net(next_states)
        next_q=V.detach().unsqueeze(-1)
        tmp = ((1 - dones) * self.gamma ** lengths)
        expected_q = rewards + tmp * next_q

        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

class BNAFInnerModel(NAFInnerModel):
    def _build_policy_output_net(self):
        output_net = BNAFOutput(64, self.action_space.shape[0],
                                      a_min=self.action_space.low, a_max=self.action_space.high)
        return output_net
class PopArtNAFInnerModel(NAFInnerModel):
    def _build_policy_output_net(self):
        output_net = NAFPopArtOutput(64, self.action_space.shape[0],
                                      a_min=self.action_space.low, a_max=self.action_space.high)
        return output_net
    def get_critic_loss(self, data, weights, update_stats=True,return_only_loss=True):
        current_q = self.get_current_value(data["state"], data["action"], to_basic_type=False)
        with torch.no_grad():
            expected_q = self.get_target_value(data["next_state"], data["reward"], data["done"], data["length"],
                                               to_basic_type=False)
        weights = torch.asarray(weights, device=self.device)
        weights = weights.unsqueeze(1)
        popart = self.policy_net.output_net.popart
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
        td_loss = (weights * nn.SmoothL1Loss(reduction='none')(u_norm, y_norm)).mean()
        return td_loss if return_only_loss else (td_loss, current_q, expected_q)
class ConservativeDoubleNAFInnerModel(NAFInnerModel):
    #policy_net()的输出V,Q为[B,]还没调整
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.ema_max=None
        self.ema_max1 = None
        self.ema_cql_loss=None

        self.alpha_min = 1e-4
        self.alpha_max = 10
        self.target_ratio = 0.1  # 想让 cons_loss ≈ 0.1 * td_loss

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.scheduler= CosineAnnealingLR(self.optimizer, T_max=200*2000, eta_min=1e-6)

    def predict(self, state):
        self.policy_net.eval()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mu,V,Q,P = self.policy_net(state,index=0)
        self.policy_net.train()
        return mu,V,Q,P

    def _build_policy_output_net(self):
        output_net1 = NAFPopArtOutput(64, self.action_space.shape[0],
                               a_min=self.action_space.low,a_max=self.action_space.high)
        output_net2 = NAFPopArtOutput(64, self.action_space.shape[0],
                               a_min=self.action_space.low,a_max=self.action_space.high)
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

        loss.backward()
        #效果反而会差，在第400次更新后会突然梯度爆炸
        #torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)
        if self.tensorboard_log_dir is not None:
            self.summary_writer.add_scalar(self.base_tag+"/loss/critic", critic_loss.detach().cpu().numpy().item(), self.update_cnt)
            #self.summary_writer.add_scalar("loss/ema_max/value", self.ema_max,
                                  #         self.update_cnt)
            #self.summary_writer.add_scalar("loss/cql/value", cql_loss.detach().cpu().numpy().item(), self.update_cnt)
            #self.summary_writer.add_scalar("loss/cql/alpha", self.alpha, self.update_cnt)
            self.summary_writer.add_scalar(self.base_tag+"/loss/total", loss.detach().cpu().numpy().item(), self.update_cnt)

            # self.writer.add_scalar("current_q",current_q.detach().cpu().numpy().item(), self.update_cnt)
            current = current_q.detach().cpu().numpy().mean().item()
            target = expected_q.detach().cpu().numpy().mean().item()
            td_abs = abs(current - target)
            self.summary_writer.add_scalar("Q/current_mean", current, self.update_cnt)
            self.summary_writer.add_scalar("Q/target_mean", target, self.update_cnt)
            self.summary_writer.add_scalar("TD/abs_mean", td_abs, self.update_cnt)
            self.summary_writer.add_scalar("lr/scheduler", self.optimizer.param_groups[0]['lr'],
                                           self.update_cnt)

            self.summary_writer.add_scalars("Q/current_and_target", {"current": current, "target": target},
                                            self.update_cnt)
        self.optimizer.step()
        self.scheduler.step()
    def get_current_value(self, states,actions,to_basic_type,net=0):

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
        rewards=rewards.unsqueeze(-1)
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
    '''
    该改进逻辑有问题,结果很差需要修改
    '''
    def get_target_value(self, next_states,rewards,dones,lengths, to_basic_type):
        # [batch_size,action_dim]
        mu1,V1,_,_=self.policy_net(next_states)
        mu, V, Q,_ = self.target_net(next_states,mu1)
        next_q=Q.detach().unsqueeze(-1)
        tmp = ((1 - dones) * self.gamma ** lengths)
        expected_q = rewards + tmp * next_q
        if to_basic_type:
            return expected_q.detach().cpu().numpy()
        return expected_q

class TimeSeriesNAFInnerModel(RLTimeSeriesInnerModelDecorator(NAFInnerModel)):
    def _build_policy_input_net(self):
        if isinstance(self.state_space,int):
            # def __init__(self, input_size, hidden_size=64, output_size=1,dropout_rate=0.5):
            return LSTMInput(input_size=self.state_space, hidden_size=128,output_size=64,dropout_rate=0.3)
        elif isinstance(self.state_space, (gymnasium.spaces.box.Box)):
            if len(self.state_space.shape) == 1:
                return LSTMInput(input_size=self.state_space.shape[0], hidden_size=128,output_size=64,dropout_rate=0.3)

#还未修改
class MAFExplorationStrategy(ContinuousExplorationStrategy):
    __exclude__ = [torch.utils.tensorboard.writer.SummaryWriter]
    def __deserialize_post__(self):
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
    def __init__(self, a_max,a_min,eps_start=1,eps_min=0.01,T=200000,tensorboard_log_dir=None):
        self.a_max = a_max
        self.a_min = a_min
        self.eps_start = eps_start
        self.eps_min =eps_min
        self.T=T
        self.decay_rate_step = (eps_min / eps_start) ** (1.0 / T)
        self.cnt=0
    def reset(self):
        pass
    def select_action(self, mean_action,*args,step=None,is_eval=False,episode=None,**kwargs):
        a_max = torch.from_numpy(self.a_max).to(mean_action.device)
        a_min = torch.from_numpy(self.a_min).to(mean_action.device)
        if is_eval:
            action= mean_action
            action = torch.max(torch.min(action, a_max), a_min)
            return action.detach().cpu().numpy()

        eps = max(self.eps_min, self.eps_start * (self.decay_rate_step ** step))
        sigma = eps * (a_max - a_min) / 2
        # action = mu + sigma * torch.randn_like(mu)
        if step % 100 == 0:
            print("sigma:", sigma,"mu:",mean_action)
        P=args[2]
        Lc = torch.linalg.cholesky(P)  # Lc · Lcᵀ = P
        Sigma_raw = torch.cholesky_inverse(Lc)
        # 强制对称：
        Sigma = 0.5 * (Sigma_raw + Sigma_raw.transpose(-1, -2))  # Σ = P⁻¹

        dist = MultivariateNormal(mean_action, Sigma)
        # dist = Normal(action_value.squeeze(-1), 1)
        action = sigma * dist.sample()

        if self.tensorboard_log_dir is not None:
            with torch.no_grad():
                delta=torch.abs(action-mean_action)
                self.summary_writer.add_scalar('NAFExploration/delta/mean',delta.mean() , self.cnt)
                self.summary_writer.add_scalar('NAFExploration/delta/max', delta.max(), self.cnt)
        #action=dist.sample()

        #action=mean_action+0.2*sigma*torch.randn_like(mean_action)

        action = torch.max(torch.min(action, a_max), a_min)
        self.cnt=self.cnt+1
        return action.detach().cpu().numpy()[0]
class MAFInnerModel(DQNInnerModel):
    support_state_types = [gymnasium.spaces.Box, int]
    support_action_types = [gymnasium.spaces.Box]

    def get_default_continuous_exploration_strategy(self):
        if self.tensorboard_log_dir == None:
            tensorboard_log_dir = None
        else:
            tensorboard_log_dir = os.path.join(self.tensorboard_log_dir, "NAFExploration")
        return NAFExplorationStrategy(a_max=self.action_space.high, a_min=self.action_space.low, eps_min=0.01,
                                      eps_start=1, T=1000000, tensorboard_log_dir=tensorboard_log_dir)

    def predict(self, state):
        self.policy_net.eval()
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mus,Ps,w,V,Q = self.policy_net(state)
        self.policy_net.train()
        return mus, Ps, w, V, Q


    def _select_action(self, state,is_eval=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mus,Ps,w,_,_ = self.policy_net(state)
        # Mixture-of-Gaussians 抽样
            cat = Categorical(w)
            samples = []
            a_max = torch.from_numpy(self.action_space.high).to(self.device)
            a_min = torch.from_numpy(self.action_space.low).to(self.device)
            # 选取 weight 最大的模式
            k_star = w.argmax(dim=1).item()  # int in [0, K)
            mu = mus[0, k_star]  # [D]
            P=Ps[0,k_star]
        if is_eval:
            action=mu
        else:
            '''
            lambda_min=1
            lambda_max=100
            P = clamp_covariance(P, lambda_min, lambda_max)
            '''


            Lc = torch.linalg.cholesky(P)  # Lc · Lcᵀ = P
            Sigma_raw = torch.cholesky_inverse(Lc)

            # 强制对称：
            Sigma = 0.5 * (Sigma_raw + Sigma_raw.transpose(-1, -2))  # Σ = P⁻¹
            dist=MultivariateNormal(mu, covariance_matrix=Sigma)
            action=dist.sample()
            action = torch.max(torch.min(action, a_max), a_min)  # [action_dim]

        '''
        n_samples=10
        start_time=time.time()
        for _ in range(n_samples):
            k = cat.sample()
            dist = MultivariateNormal(mus[:,k,:], covariance_matrix=Ps[:,k,:])
            a = dist.sample()  # [1,action_dim]
            a = torch.max(torch.min(a, a_max), a_min)[0]  # [action_dim]
            a=a.squeeze(0)
            samples.append(a)
        
        # 评估 Q
        states = state.expand(n_samples, -1)  # [num_samples, state_dim]
        actions = torch.stack(samples, dim=0)  # [num_samples, action_dim]
        _, _,_,_,Q_vals = self.policy_net(states, actions)  # [num_samples]
        # 取最优
        idx = Q_vals.argmax().item()
        action = samples[idx]
        '''
        return action.detach().cpu().numpy()


    def _build_policy_output_net(self):
        output_net = MAFOutput(64, self.action_space.shape[0],
                               a_min=self.action_space.low, a_max=self.action_space.high)
        return output_net

    def get_max_action_value(self, states, to_basic_type=True):
        _,Q=self.target_net(states)
        if to_basic_type:
            return Q.detach().cpu().numpy()
        return Q

    def get_current_value(self, data, to_basic_type):
        #1ms
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

        _,_,_,_,Q = self.policy_net(states, actions)
        current_q = Q.unsqueeze(-1)

        if flag:
            # action如果是一个值则[batch_size,新增action_shape=1,]
            current_q = current_q.unsqueeze(1)

        if to_basic_type:
            return current_q.detach().cpu().numpy()
        return current_q

    def get_target_value(self, data, to_basic_type):
        #3ms
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
        rewards = rewards.unsqueeze(-1)
        actions = torch.stack(data1["action"])
        batch_size=states.shape[0]
        mus,_,w,_,_=self.target_net(next_states)
        idx = torch.argmax(w, dim=1)  # [B]
        mu = mus[torch.arange(batch_size), idx]  # [B, D]
        _,_,_,_,next_q=self.target_net(next_states, mu)
        '''
        B,K,D=mus.shape
        mus_flat = mus.view(B * K, D)
        states_flat = next_states.unsqueeze(1)  # [B, 1, S]
        states_flat = states_flat.expand(B, K, -1)  # [B, K, S]
        states_flat = states_flat.contiguous()  # 确保内存是连续的
        states_flat = states_flat.view(B * K, -1)  # [B*K, S]
        _,_,_,_,q_next_flat=self.target_net(states_flat,mus_flat)
        q_next = q_next_flat.view(B, K)  # [B, K]
        q_next_max, _ = q_next.max(dim=1)  # [B]
        next_q = q_next_max.detach()
        '''
        tmp = ((1 - dones) * self.gamma ** lengths).unsqueeze(1)
        next_q=next_q.unsqueeze(-1)
        expected_q = rewards + tmp * next_q

        if to_basic_type:
            return expected_q.detach().cpu().numpy()

        return expected_q
