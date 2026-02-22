import os
import time

import tensorboardX
import torch.utils.tensorboard.writer

from torch import nn
from torch.utils.tensorboard import SummaryWriter

from RL.common.utils.decorator.TorchMeta import SerializeAndTBMeta


class BaseNet(nn.Module,metaclass=SerializeAndTBMeta):

    def __init__(self,name=None):
        super().__init__()
        if name is None:
            self.name=self.__class__.__name__+"_"+str(id(self))
        else:
            self.name=name



class CombineNet(BaseNet):
    def __init__(self, input_net, output_net,name=None):
        super().__init__(name=name)

        if input_net.name is None:
            input_net.name=self.name+"_input_net"
        if output_net.name is None:
            output_net.name=self.name+"_output_net"
        self.input_net = input_net
        self.output_net =output_net
    def forward(self,x,*args,**kwargs):
        feat,args,kwargs=self.input_net(x,*args,**kwargs)

        return self.output_net(feat,*args,**kwargs)
    def _setup_tensorboard(self, tensorboard_log_dir=None, base_tag=None):
        super()._setup_tensorboard(tensorboard_log_dir,base_tag)
        if self.input_net.tensorboard_log_dir is None:
            self.input_net.tensorboard_log_dir =self.tensorboard_log_dir
            self.input_net.summary_writer=self.summary_writer
            self.input_net.base_tag=self.base_tag+"/input_net"
        if self.output_net.tensorboard_log_dir is None:
            self.output_net.tensorboard_log_dir =self.tensorboard_log_dir
            self.output_net.summary_writer = self.summary_writer
            self.output_net.base_tag=self.base_tag+"/output_net"
class EnsembleNet(BaseNet):
    def __init__(self, *submodules: nn.Module,name=None):
        super().__init__(name=name)
        self.models = nn.ModuleList(submodules)
        '''
        没有处理tensorboard_log_dir
        '''
    def forward(self, x,*args,index=-1,**kwargs):
        if index==-1 or not isinstance(index,int) or index>=len(self.models):
            return [m(x,*args,**kwargs) for m in self.models]
        else:
            return self.models[index](x,*args,**kwargs)
    def __getitem__(self, item):
        return self.models[item]
