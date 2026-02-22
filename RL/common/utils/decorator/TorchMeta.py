import functools
import os
from functools import wraps

import torch
from torch.utils.tensorboard import SummaryWriter

from RL.common.utils.decorator.Meta import SerializeMeta


class TBMeta(type):
    '''
    新增setup_tensorboard方法，在tensorboard_log_dir不为None时生成summary_writer，否则设置summary_writer=None
    对类新增tensorboard_log_dir属性，并且传入setup_tensorboard参数调用setup_tensorboard方法
    '''

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        orig_init = namespace.get('__init__', None)
        if orig_init is None:
            for base in bases:
                parent_init = getattr(base, '__init__', None)
                if parent_init is not None:
                    orig_init = parent_init
                    break
        # 如果还没找到，就落到 object.__init__
        orig_init = orig_init or object.__init__
        setup_tensorboard = namespace.get('setup_tensorboard', None)
        if setup_tensorboard is None:
            for base in bases:
                parent_setup = getattr(base, 'setup_tensorboard', None)
                flag=getattr(base,'is_setup_tensorboard_injected', False)
                if parent_setup is not None and not flag:
                    setup_tensorboard = parent_setup
                    break
        _setup_tensorboard = namespace.get('_setup_tensorboard', None)
        if _setup_tensorboard is None:
            for base in bases:
                parent_setup = getattr(base, '_setup_tensorboard', None)
                if parent_setup is not None:
                    _setup_tensorboard = parent_setup
                    break

        # 注入统一的设置方法
        def setup_tensorboard_inject(self, tensorboard_log_dir=None,base_tag=None,run_=True):
            """
            设置 tensorboard_log_dir 并初始化 SummaryWriter
            """
            self.tensorboard_log_dir = tensorboard_log_dir
            if base_tag is not None:
                self.base_tag = base_tag
            else:
                self.base_tag = self.__class__.__name__
            if tensorboard_log_dir:
                os.makedirs(tensorboard_log_dir, exist_ok=True)
                self.summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)
            else:
                self.summary_writer = None
            if run_==True and self.__class__==cls and self.tensorboard_log_dir is not None:
                self._setup_tensorboard(tensorboard_log_dir,base_tag)
        def _setup_tensorboard_inject(self,tensorboard_log_dir=None,base_tag=None):
            pass
        if setup_tensorboard is None:
            setup_tensorboard=setup_tensorboard_inject
            cls.is_setup_tensorboard_injected=True
        if _setup_tensorboard is None:
            _setup_tensorboard=_setup_tensorboard_inject
        cls.setup_tensorboard = setup_tensorboard
        cls._setup_tensorboard = _setup_tensorboard

        # 构造新的 __init__，确保父类 __init__ 也会被调用
        @wraps(orig_init)
        def __init__(self, *args, tensorboard_log_dir=None, **kwargs):
            # 1. 决定调用哪个 __init__：优先用自己定义的 orig_init，
            #    否则调用 MRO 上的下一个 __init__
            # 2. 初始化 TensorBoard 相关
            #
            self.setup_tensorboard(tensorboard_log_dir, run_=False)
            if orig_init is not None:
                # 如果目标类自己定义了 __init__，调用它
                orig_init(self, *args, **kwargs)
            if self.tensorboard_log_dir is not None:
                self._setup_tensorboard(tensorboard_log_dir)

        cls.__init__ = __init__
        return cls


class SerializeAndTBMeta(SerializeMeta, TBMeta):
    """
    序列化时自动排除SummaryWriter对象
    """
    def __new__(mcs, name, bases, namespace):
        # Use MRO-based super to ensure both SerializeMeta and TBMeta __new__ are called
        cls = super().__new__(mcs, name, bases, namespace)
        # Ensure SummaryWriter is excluded from serialization
        writer_cls = torch.utils.tensorboard.writer.SummaryWriter
        if not hasattr(cls, '__exclude__') or cls.__exclude__ is None:
            cls.__exclude__ = [writer_cls]
        else:
            # Avoid duplicate entries
            if writer_cls not in cls.__exclude__:
                cls.__exclude__.append(writer_cls)
        func=None
        if hasattr(cls,"__deserialize_post__"):
            func=cls.__deserialize_post__
        # Inject post-deserialization hook
        def __deserialize_post__(self):
            tb_dir = getattr(self, 'tensorboard_log_dir', None)
            base_tag=getattr(self, 'base_tag', None)
            self.setup_tensorboard(tb_dir,base_tag)
            if func is not None:
                func(self)
        cls.__deserialize_post__ = __deserialize_post__
        return cls
