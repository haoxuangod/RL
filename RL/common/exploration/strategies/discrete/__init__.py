from abc import abstractmethod

import numpy as np
import torch

from RL.common.exploration import RLExplorationStrategy


class DiscreteExplorationStrategy(RLExplorationStrategy):
    '''
    输出离散的action
    '''

    @abstractmethod
    def select_action(self, q_values: torch.Tensor, *args,step: int = None,episode=None,is_eval=False,**kwargs)->int:
        """
        输入：Q值张量（形状 [action_dim]）
        输出：选择的动作索引
        """
        raise NotImplementedError

class Greedy(DiscreteExplorationStrategy):
    '''
    选择使得评估值最大的那个action
    '''
    def select_action(self, q_values,*args,step=None,episode=None,is_eval=False,**kwargs):
        if len(q_values.shape) == 1:
            return q_values.argmax().detach().cpu().numpy().item()  # 利用
        else:
            return q_values.argmax(dim=-1).detach().cpu().numpy()
class EpsilonGreedy(DiscreteExplorationStrategy):
    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=200):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def select_action(self, q_values, *args,step=None,episode=None,is_eval=False,**kwargs):
        self.steps_done = step if step is not None else self.steps_done + 1
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-1. * self.steps_done / self.eps_decay)
        if np.random.rand() > eps or is_eval:
            if len(q_values.shape)==1:
                return q_values.argmax().detach().cpu().numpy().item()  # 利用
            else:
                return q_values.argmax(dim=-1).detach().cpu().numpy()
        else:
            if len(q_values.shape)==1:
                return np.random.randint(q_values.shape[0])  # 探索
            else:
                return np.random.randint(q_values.shape[-1],size=q_values.shape[:-1])

class Boltzmann(DiscreteExplorationStrategy):
    '''
    待修改
    '''
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def select_action(self, q_values, *args,step=None,episode=None,is_eval=False,**kwargs):
        probs = torch.softmax(q_values / self.temperature, dim=0).numpy()
        return np.random.choice(len(probs), p=probs)


class UCB(DiscreteExplorationStrategy):
    '''
        待修改
    '''
    def __init__(self, c=2.0):
        self.c = c
        self.action_counts = None  # 需在首次调用时初始化

    def select_action(self, q_values, *args,is_eval=False,**kwargs):
        if self.action_counts is None:
            self.action_counts = np.zeros(q_values.shape[0])

        total = np.sum(self.action_counts)
        ucb_values = q_values.numpy() + self.c * np.sqrt(np.log(total + 1) / (self.action_counts + 1e-5))
        action = np.argmax(ucb_values)
        self.action_counts[action] += 1
        return action