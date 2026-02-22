import gym
import gymnasium
from torch import nn

from RL.HierarchicalRL.HRLBases import HRLNodeInnerModelDecorator, HRLNode

import torch

from RL.RLBases import RLInnerModelBase
from RL.RLModels.ValueOffPolicyInnerModel.DQNModel import DQNInnerModel, NAFInnerModel, DoubleDQNInnerModel, \
    DoubleNAFInnerModel
from RL.TorchModels.BaseNet import CombineNet
from RL.TorchModels.DQN import DQNOutput
from RL.TorchModels.ModelInput import MLPInput, CNNInput


class IntraOptionNodeInnerModelDecorator(HRLNodeInnerModelDecorator):

    @classmethod
    def Decorator(cls_decorator,cls):
        class NewClass(cls):
            def __init__(self,*args,**kwargs):
                if not hasattr(cls,"get_target_value") or not hasattr(cls,"get_current_value"):
                    '''
                    没有critic网络的只基于策略新增一个默认critic网络
                    '''

                super().__init__(*args,**kwargs)
                self.termination_net=self.get_termination_net()
                self.is_intraoption_model=True

            def get_termination_net(self):
                '''
                返回计算beta(state)网络
                '''
                if hasattr(cls,"get_termination_net"):
                    return cls.get_termination_net(self)
                else:
                    raise NotImplementedError
            def get_beta(self,state,to_basic_type=False):
                '''
                当前状态终止概率
                :param state:
                '''
                if to_basic_type:
                    return torch.sigmoid(self.termination_net(state)).detach().cpu()
                return torch.sigmoid(self.termination_net(state))
            def get_termination_loss(self,data):
                '''
                直接返回torch类型的termination_loss
                '''
                if self.parent_model is None:
                    '''
                    root节点直接返回0
                    '''
                    return 0
                if isinstance(data, dict):
                    next_states = [data["next_state"]]
                else:
                    next_states = [d["next_state"] for d in data]
                next_states=torch.stack(next_states)

                fa_next_q_max = self.parent_model.get_max_action_value(next_states, to_basic_type=False).detach()

                beta=self.get_beta(next_states, to_basic_type=False)
                next_q = self.get_max_action_value(next_states, to_basic_type=False)
                adv = (next_q - fa_next_q_max).detach()
                # 终止损失：E[β * adv]
                term_loss = torch.mean(beta * adv)
                return term_loss
            def get_critic_loss(self,data,weights,return_only_loss=True):
                current_q = self.get_current_value(data, to_basic_type=False)
                expected_q = self.get_target_value(data, to_basic_type=False)
                weights = torch.asarray(weights, device=self.device)
                weights = weights.unsqueeze(1)
                # reduction='none'表示不对batch平方差求平均
                loss = (weights * nn.MSELoss(reduction='none')(current_q, expected_q)).mean()
                if return_only_loss:
                    return loss
                else:
                    return loss, current_q, expected_q
            def _update(self, indices, batch_data, weights):
                '''
                loss=IntraOptionLoss+term_loss+policy_loss(如果有的话)
                '''
                critic_loss, current_q, expected_q = self.get_critic_loss(batch_data, weights, return_only_loss=False)
                term_loss=self.get_termination_loss(batch_data)
                if hasattr(self,"get_actor_loss"):
                    actor_loss = self.get_actor_loss(batch_data)
                    loss = term_loss + critic_loss+actor_loss
                else:
                    loss=term_loss+critic_loss
                self.memory.update(indices, current_q.detach().cpu().numpy(), expected_q.detach().cpu().numpy())
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            def get_target_value(self, data, to_basic_type):
                # [batch_size,action_dim]
                if isinstance(data, dict):
                    next_states=[data["next_state"]]
                else:
                    next_states = [d["next_state"] for d in data]
                next_states = torch.stack(next_states)

                beta=self.get_beta(next_states)
                if self.parent_model is None:
                    expected_q = super().get_target_value(data, to_basic_type=False)
                else:
                    expected_q = ((1-beta)*super().get_target_value(data, to_basic_type=False)+
                                  beta*self.parent_model.get_target_value(data,to_basic_type=False)).detach()

                if to_basic_type:
                    return expected_q.detach().cpu().numpy()
                return expected_q
            def get_best_model(self):
                best_model = super().get_best_model()
                best_model["termination_net"] = self.termination_net.state_dict()
                return best_model
            def load_best_model(self):
                if not len(self.best_model)==0:
                    super().load_best_model()
                    self.termination_net.load_state_dict(self.best_model["termination_net"])

        return NewClass

    def __new__(cls_decorator, cls):
        return cls_decorator.Decorator(super().Decorator(cls))


class IntraOptionDQNInnerModel(IntraOptionNodeInnerModelDecorator(DQNInnerModel)):
    def get_termination_net(self):
        output_net = DQNOutput(64, 1)
        if isinstance(self.state_space, int):
            input_net = MLPInput(self.state_shape, 64)
        elif isinstance(self.state_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            if len(self.state_space.shape) == 1:
                input_net = MLPInput(self.state_space.shape, 64)

            elif len(self.state_space.shape) == 2 or len(self.state_space.shape) == 3:
                input_net = CNNInput(self.state_space.shape, 64).to(self.device)
            else:
                raise ValueError("state_shape must be int or tuple/list with length 1、2、3")
        else:
            raise ValueError("state_shape must be int or tuple/list with length 1、2、3")

        return CombineNet(input_net, output_net).to(self.device)

class IntraOptionDoubleDQNInnerModel(IntraOptionNodeInnerModelDecorator(DoubleDQNInnerModel)):
    def get_termination_net(self):
        output_net = DQNOutput(64, 1)
        if isinstance(self.state_space, int):
            input_net = MLPInput(self.state_shape, 64)
        elif isinstance(self.state_space, (gymnasium.spaces.box.Box, gym.spaces.box.Box)):
            if len(self.state_space.shape) == 1:
                input_net = MLPInput(self.state_space.shape, 64)

            elif len(self.state_space.shape) == 2 or len(self.state_space.shape) == 3:
                input_net = CNNInput(self.state_space.shape, 64).to(self.device)
            else:
                raise ValueError("state_shape must be int or tuple/list with length 1、2、3")
        else:
            raise ValueError("state_shape must be int or tuple/list with length 1、2、3")

        return CombineNet(input_net, output_net).to(self.device)

class IntraOptionNAFInnerModel(IntraOptionNodeInnerModelDecorator(NAFInnerModel)):
    def get_termination_net(self):
        input_net = None
        # 判断action_space
        output_net = DQNOutput(64, 1)
        if isinstance(self.state_space, int):
            input_net = MLPInput(self.state_space, 64)
        else:
            if len(self.state_space.shape) == 1:
                input_net = MLPInput(self.state_space.shape, 64)
            elif len(self.state_space.shape) == 2 or len(self.state_space.shape) == 3:
                input_net = CNNInput(self.state_space.shape, 64).to(self.device)
            else:
                raise ValueError("state_shape must be Box with the length of shape： 1、2、3")

        return CombineNet(input_net, output_net).to(self.device)

class IntraOptionDoubleNAFInnerModel(IntraOptionNodeInnerModelDecorator(DoubleNAFInnerModel)):
    def get_termination_net(self):
        input_net = None
        # 判断action_space
        output_net = DQNOutput(64, 1)
        if isinstance(self.state_space, int):
            input_net = MLPInput(self.state_space, 64)
        else:
            if len(self.state_space.shape) == 1:
                input_net = MLPInput(self.state_space.shape, 64)
            elif len(self.state_space.shape) == 2 or len(self.state_space.shape) == 3:
                input_net = CNNInput(self.state_space.shape, 64).to(self.device)
            else:
                raise ValueError("state_shape must be Box with the length of shape： 1、2、3")

        return CombineNet(input_net, output_net).to(self.device)
class IntraOptionNode(HRLNode):
    '''
    设置模型的is_primitive属性（如果有），表示当前节点为原子任务节点
    '''
    columns=["state","action","reward","next_state","done","length"]
    def __init__(self, name, model:RLInnerModelBase,parent=None):
        super().__init__(name=name, model=model, parent=parent)
        if not (hasattr(model,"is_intraoption_model") and model.is_intraoption_model==True):
            raise ValueError("the model is not an IntraOption model which must use IntraOptionNodeInnerModelDecorator")

    def is_terminal(self, state):
        #需要优化
        state = torch.tensor(state, device=self.model.device,dtype=torch.float32)
        beta=self.model.get_beta(state).squeeze()
        return torch.rand_like(beta)<=beta