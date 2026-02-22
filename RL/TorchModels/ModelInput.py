import operator
import time
from functools import reduce

import torch
from torch import nn

from RL.TorchModels.BaseNet import BaseNet


class MLPInput(BaseNet):
    """
    input_shape: int 或 长度为1的 list/tuple
    output_shape: int 或 tuple 或 list
    """
    def __init__(self, input_shape, output_shape,name=None):
        super().__init__(name=name)
        # 处理 input_shape
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            self.input_shape = input_shape[0]
        elif isinstance(input_shape, int):
            self.input_shape = input_shape
        else:
            raise ValueError(f"input_shape must be int or length-1 list/tuple, but got {input_shape}")

        # 保存原始 output_shape 以便 reshape
        if isinstance(output_shape, int):
            self.output_shape = (output_shape,)
        elif isinstance(output_shape, (list, tuple)):
            self.output_shape = tuple(output_shape)
        else:
            raise ValueError(f"output_shape must be int, list or tuple, but got {output_shape}")

        # 特征提取层
        self.fc = nn.Sequential(
            nn.Linear(self.input_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # 自动获取 fc 的输出维度
        last_linear = None
        # 倒序找第一个 Linear
        for module in reversed(self.fc):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        feature_dim = last_linear.out_features

        # 计算最终输出单元数
        out_units = reduce(operator.mul, self.output_shape, 1)

        # 最后一层 mapping 到用户指定的 output_shape
        self.head = nn.Linear(feature_dim, out_units)

    def forward(self, x,*args,**kwargs):
        # x: (B, self.input_shape)
        #这行有什么用?
        #torch.cuda.synchronize()
        feat = self.fc(x)
        out = self.head(feat)        # (B, ∏output_shape)
        # 如果 output_shape 长度>1，则 reshape
        if len(self.output_shape) > 1:
            out = out.view(x.size(0), *self.output_shape)
        else:
            # 如果只是单个维度, 返回 (B,) 或 (B,1) 可根据需求改成 squeeze
            out = out.squeeze(-1)
        return out,args,kwargs


class CNNInput(BaseNet):
    """
    input_shape: 2-tuple (w,c) 或 3-tuple (h,w,c)
    output_shape: int 或 tuple 或 list
    """
    def __init__(self, input_shape, output_shape,name=None):
        super().__init__(name=name)
        # 处理 input_shape
        if len(input_shape) == 2:
            h = 1
            w, c = input_shape
        elif len(input_shape) == 3:
            h, w, c = input_shape
        else:
            raise ValueError(f"input_shape must be 2-tuple (w,c) or 3-tuple (h,w,c), got {input_shape}")

        # 保存 raw output_shape
        if isinstance(output_shape, int):
            self.output_shape = (output_shape,)
        elif isinstance(output_shape, (list, tuple)):
            self.output_shape = tuple(output_shape)
        else:
            raise ValueError(f"output_shape must be int, list or tuple, but got {output_shape}")
        self.input_shape = input_shape
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        # 计算 conv 输出尺寸
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_w = conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2)
        conv_h = conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2)
        linear_input_size = conv_w * conv_h * 32

        # 特征层
        self.feature = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.ReLU(),
        )
        feature_dim = self.feature[0].out_features  # 256

        # 最后 head
        out_units = reduce(operator.mul, self.output_shape, 1)
        self.head = nn.Linear(feature_dim, out_units)

    def forward(self, x):
        # x: (H,W,C) 或 (B,H,W,C)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # 转 (B,C,H,W)
        x = x.permute(0, 3, 1, 2).float()
        # 卷积 + 展平
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        feat = self.feature(x)
        out = self.head(feat)
        # reshape 到期望形状
        if len(self.output_shape) > 1:
            out = out.view(x.size(0), *self.output_shape)
        else:
            out = out.squeeze(-1)
        return out

class LSTMInput(BaseNet):
    def __init__(self, input_size, name=None,hidden_size=64, output_size=1,dropout_rate=0.5):
        super().__init__(name=name)
        # 第一层LSTM：输入特征数由数据决定，输出所有时间步
        self.lstm1 = nn.LSTM(
            input_size=input_size,  # 输入特征维度（需根据数据集调整）
            hidden_size=hidden_size,
            batch_first=True
        )
        # 第二层LSTM：自动获取前一层输出维度，仅返回最后时间步
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Dropout层（默认概率0.5）
        self.dropout = nn.Dropout(dropout_rate)
        '''
        # 全连接输出层
        self.dense  = nn.Sequential(
        nn.Linear(hidden_size, 1),
        nn.ReLU()
        )
        '''
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 输入形状：(batch_size, seq_length, input_size)
        x, _ = self.lstm1(x)  # 输出形状：(batch, 30, 30)
        #x, (hn, cn) = self.lstm2(x)  # 输出形状：(batch, 30, 30)

        # 取序列最后一个时间步的输出
        x = x[:, -1, :]  # 输出形状：(batch, 30)

        x = self.dropout(x)
        x = self.dense(x)  # 输出形状：(batch, output_size)
        return x