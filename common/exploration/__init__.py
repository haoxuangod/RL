
from abc import abstractmethod

import torch

from RL.common.schedules import Schedule
from RL.common.utils.decorator.TorchMeta import SerializeAndTBMeta


class RLExplorationStrategy(metaclass=SerializeAndTBMeta):

    def __init__(self,schedule:Schedule=None):
        if schedule is not None:
            self.schedule =schedule
        else:
            self.schedule =self.get_default_schedule()
    def get_default_schedule(self):
        return None

    @abstractmethod
    def select_action(self, q_values: torch.Tensor, step: int = None,is_eval=False):
        """
        输入：Q值张量（形状 [action_dim]）
        输出：选择的动作索引
        """
        pass
    def update_params(self, *args, **kwargs):
        """可选：用于更新策略内部参数（如ε衰减）"""
        pass
    def reset(self):
        '''
        每回合结束后调用
        '''