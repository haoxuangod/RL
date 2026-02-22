import numpy as np
import tensorboardX
import torch
from torch.utils.tensorboard import SummaryWriter

from RL.common.exploration import RLExplorationStrategy
from RL.common.exploration.strategies.continuous.noises.gaussian_noise import GaussianNoise
from RL.common.exploration.strategies.continuous.noises.ou_noise import OUNoise
from RL.common.schedules import Schedule
from RL.common.schedules.schedules import LinearSchedule


class ContinuousExplorationStrategy(RLExplorationStrategy):
    def __init__(self,action_dim,a_min,a_max,schedule:Schedule=None,schedule_mode="step",noise=None):
        super().__init__(schedule=schedule)
        self.schedule_mode = schedule_mode
        self.a_min = a_min
        self.a_max = a_max
        self.action_dim = action_dim
        if noise is not None:
            self.noise = noise
        else:
            self.noise = self.get_default_noise()
    def get_default_noise(self):
        return GaussianNoise(sigma=0.2)
    def select_action(self, mean_action,*args,step=None,is_eval=False,episode=None,**kwargs):
        if is_eval:
            return mean_action
        else:
            if self.schedule is not None:
                if self.schedule_mode=="step":
                    rate=self.schedule(step=step)
                else:
                    rate=self.schedule(step=episode)
            else:
                rate=1
            noise=self.noise(action_dim=self.action_dim,step=step,episode=episode)
            if self.tensorboard_log_dir is not None:
                self.summary_writer.add_scalar(f'{self.base_tag}/{self.noise.__class__.__name__}/noise/mean', noise.mean(),step)
                self.summary_writer.add_scalar(f'{self.base_tag}/{self.noise.__class__.__name__}/noise/max', noise.max(), step)
                if self.schedule:
                    if self.schedule_mode=="step":
                        self.summary_writer.add_scalar(f'{self.base_tag}/schedule/value', rate, step)
                    else:
                        self.summary_writer.add_scalar(f'{self.base_tag}/schedule/value', rate, episode)

            action = mean_action.detach().cpu().numpy() + rate*noise
            action = np.minimum(np.maximum(action, self.a_min), self.a_max)

            return action
class ContinuousDecayingWeightMixture(ContinuousExplorationStrategy):
    '''
    线性因子融合两个策略select_action的输出
    action=alpha*action1+(1-alpha)*action2
    其中alpha随着时间衰减
    '''
    def __init__(self, schedule: Schedule,strategy1:ContinuousExplorationStrategy,strategy2:ContinuousExplorationStrategy,tensorboard_log_dir=None):
        super().__init__(schedule=schedule)
        self.strategy1 = strategy1
        self.strategy2 = strategy2

    def select_action(self, mean_action,*args,step=None,is_eval=False,episode=None,**kwargs):
        if is_eval:
            return mean_action
        action1 = self.strategy1.select_action(mean_action,*args,step=step,is_eval=is_eval,episode=episode)
        action2 = self.strategy2.select_action(mean_action,*args,step=step,is_eval=is_eval,episode=episode)
        alpha=self.schedule(step=step)
        action=alpha*action1+(1-alpha)*action2
        return action

'''
class GaussianNoise(ContinuousExplorationStrategy):
    def __init__(self, sigma,schedule=None):
        super().__init__(schedule=schedule)
        self.sigma = sigma
    def select_action(self, mean_action,*args,step=None,is_eval=False,episode=None,**kwargs):
        if is_eval:
            return mean_action
        else:
            return mean_action + np.random.normal(0, self.sigma, size=mean_action.shape)

class OUNoise(ContinuousExplorationStrategy):
    def __init__(self, action_dimension,mu=0.0,theta=0.15, sigma=0.2,a_min=-1,a_max=1,schedule=None,record_stats_dir=None):
        self.mu, self.theta, self.sigma = mu, theta, sigma
        super().__init__(schedule=schedule)
        self.action_dimension =action_dimension
        self.record_stats_dir=record_stats_dir
        if self.record_stats_dir is not None:
            # 创建一个 writer，指定日志保存目录
            self.summary_writer = SummaryWriter(log_dir=record_stats_dir)
        else:
            self.summary_writer = None
        self.state = None
        self.a_min = a_min
        self.a_max = a_max
        self.cnt=0
        self.reset()

    def get_default_schedule(self):
        return LinearSchedule(total_steps=10000,initial_value=1,final_value=0.1)
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        self.cnt=0
    def select_action(self, mean_action,*args,step=None,is_eval=False,episode=None,**kwargs):
        x=self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state =x+dx
        self.cnt=self.cnt+1
        if self.record_stats_dir is not None:
            self.summary_writer.add_scalar('OuNoise/state_mean',self.state.mean(),self.cnt)
            self.summary_writer.add_scalar('OuNoise/state_max', self.state.max(),self.cnt)
        noise=self.state
        if self.schedule:
            noise=self.schedule(step)*self.state
        if step%100==0:
            print("noise:",noise,"rate:",self.schedule(step))
        action = mean_action.detach().cpu().numpy()+noise
        action=np.clip(action,self.a_min,self.a_max)

        return  action
'''