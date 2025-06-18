import tensorboardX
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_

from RL.RLBases import OffPolicyInnerModel, ReplayBuffer



class ValueOffPolicyInnerModel(OffPolicyInnerModel):
    def __init__(self,state_space,action_space, memory:ReplayBuffer=None,
                 gamma=0.99,batch_size=32,memory_capacity=10000,update_freq=4,lr=1e-4,
                 steps_before_update=1,tensorboard_log_dir=None,exploration_strategy=None,state_processor=None):
        super().__init__(state_space=state_space,action_space=action_space,state_processor=state_processor,
                         memory=memory,gamma=gamma,batch_size=batch_size,memory_capacity=memory_capacity,update_freq=update_freq,
                         steps_before_update=steps_before_update,tensorboard_log_dir=tensorboard_log_dir,exploration_strategy=exploration_strategy)
        self.lr=lr

    def get_current_value(self, data, to_basic_type):
        '''
        返回data中state的预估评分
        :param data: 可以是一个dict或者是[dict1,dict2,dict3...]即批量获取或者单个获取
        :param to_basic_type:是否将结果转换为float/int等初始类型，因为torch在训练时需要
        模型的梯度信息必须返回tensor，而调用memory的append方法的current_value应该为基础类型
        :return:
        '''
        raise NotImplementedError

    def get_target_value(self, data, to_basic_type: bool):
        '''
        返回data中目标的评分(如TD误差)
        :param data: 可以是一个dict数据或者是[dict1,dict2,dict3...]即批量数据或者单个数据
        :param to_basic_type:是否将结果转换为float/int等初始类型，因为torch在训练时需要
        模型的梯度信息必须返回tensor，而调用memory的append方法的target_value应该为基础类型
        :return:
        '''
        raise NotImplementedError

    def _update(self, indices, batch_data, weights):
        # 反向传播
        self.optimizer.zero_grad()
        critic_loss,current_q,expected_q=self.get_critic_loss(batch_data,weights,return_only_loss=False)
        self.memory.update(indices, current_q.detach().cpu().numpy(), expected_q.detach().cpu().numpy())
        critic_loss.backward()
        clip_grad_norm_(self.policy_net.parameters(), 1)
        if self.tensorboard_log_dir is not None:
            self.summary_writer.add_scalar(self.base_tag+"/loss/critic", critic_loss.detach().cpu().numpy().item(), self.update_cnt)
            #self.writer.add_scalar("current_q",current_q.detach().cpu().numpy().item(), self.update_cnt)
            current=current_q.detach().cpu().numpy().mean().item()
            target= expected_q.detach().cpu().numpy().mean().item()
            td_abs=abs(current-target)
            self.summary_writer.add_scalar(self.base_tag+"/Q/current_mean", current, self.update_cnt)
            self.summary_writer.add_scalar(self.base_tag+"/Q/target_mean", target, self.update_cnt)
            self.summary_writer.add_scalar(self.base_tag+"/TD/abs_mean", td_abs, self.update_cnt)
            self.summary_writer.add_scalar(self.base_tag+"/lr/scheduler",self.optimizer.param_groups[0]['lr'],self.update_cnt)
            self.summary_writer.add_scalars(self.base_tag+"/Q/current_and_target",{"current":current,"target":target},self.update_cnt)
        self.optimizer.step()
        if hasattr(self,"scheduler"):
            self.scheduler.step()



    def get_critic_loss(self,data,weights,return_only_loss=True):
        current_q = self.get_current_value(data, to_basic_type=False)
        with torch.no_grad():
            expected_q = self.get_target_value(data, to_basic_type=False)
        '''
        print("current_q",current_q.requires_grad, current_q.grad_fn)
        print("target:", expected_q.requires_grad, expected_q.grad_fn)
        '''
        weights = torch.asarray(weights, device=self.device)
        weights = weights.unsqueeze(1)
        # reduction='none'表示不对batch平方差求平均
        #loss = (weights * nn.MSELoss(reduction='none')(current_q, expected_q)).mean()
        #loss = (weights * nn.SmoothL1Loss(reduction='none')(current_q, expected_q)).mean()
        loss=nn.SmoothL1Loss()(current_q, expected_q)

        if return_only_loss:
            return loss
        else: return loss,current_q, expected_q
