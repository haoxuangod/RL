import tensorboardX


import torch
import io

from RL.RLBases import  ReplayBuffer
from RL.common.utils.decorator.Meta import Serializer


class TorchSerializer(Serializer):
    supported_types = [torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
    def _serialize(self,obj):
        # 创建内存字节流对象
        buffer = io.BytesIO()
        torch.save(obj,buffer)
        buffer.seek(0)
        model_bytes = buffer.getvalue()  # 获取字节数据
        return str(model_bytes.hex())
    def _deserialize(self,cls,data):
        buffer = io.BytesIO(bytes.fromhex(data))
        obj=torch.load(buffer)
        return obj


from contextlib import contextmanager

'''
@contextmanager
def auto_device(device):
    original_type = torch.get_default_dtype()
    try:
        # 在上下文内新建张量自动分配设备 
        torch.set_default_ttype(torch.cuda.FloatTensor if "cuda" in str(device) else torch.FloatTensor)
        yield
    finally:
        torch.set_default_dtype(original_type)

    # 使用示例 


device = torch.device("cuda:0")
with auto_device(device):
    new_tensor = torch.randn(3, 3)  # 自动在cuda:0上创建 
'''
class TorchRLInnerModelDecorator:
    '''
    后续可添加自动化对所有torch中元素使用.to(device)
    '''
    @classmethod
    def Decorator(cls_decorator,cls):
        class NewClass(cls):
            serializers = [TorchSerializer()]

            def __init__(self,*args,device=None,**kwargs):
                super().__init__(*args,**kwargs)
                if device is None:
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    self.device =device
            '''
            def convert_data(self,key,value):
                if key=="state":
                    return torch.tensor(value, device=self.device,dtype=torch.float32)
                elif key=="reward":
                    return torch.tensor(value, device=self.device,dtype=torch.float32)
                elif key=="next_state":
                    return torch.tensor(value, device=self.device,dtype=torch.float32)
                elif key=="done":
                    return torch.tensor(value,dtype=torch.int32,device=self.device)
                elif key=="length":
                    return torch.tensor(value, device=self.device)
                elif key=="action":
                    #需要修改
                    return torch.tensor(value, dtype=torch.float32, device=self.device)
                    #return torch.tensor(value, dtype=torch.int64,device=self.device)
                else:
                    return value
            '''
            def append_memory(self,data,episode):
                self.memory.append(data,episode,self)
        return NewClass
    def __new__(cls_decorator,cls):
        return cls_decorator.Decorator(cls)


