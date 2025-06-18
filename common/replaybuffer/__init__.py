import random
from collections import deque

import numpy
import numpy as np

from RL.common.utils.decorator.Meta import SerializeMeta


class InnerBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.tot_add = 0
        self.last_episode = -1
        # 记录每个episode的初始位置和结束位置
        self.episode_dict = {}
        self._new_episode_flag = True

    def __len__(self):
        return self.memory.__len__()

    def __getitem__(self, item):
        return self.memory[item]

    def update(self, indices, current_value, target_value):
        # 模型运行后更新后会调用buffer的更新如优先级采样回放区
        pass

    def append(self, data, episode, inner_model=None):
        self.tot_add = self.tot_add + 1
        if episode != self.last_episode:
            self.episode_dict[episode] = (self.tot_add, self.tot_add)
            self.last_episode = episode
            self._new_episode_flag = True
        else:
            tmp = self.episode_dict[episode]
            self.episode_dict[episode] = (tmp[0], self.tot_add)
        self._append(data, episode, inner_model)

    def _append(self, data, episode, inner_model=None):
        self.memory.append(data)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.memory)), batch_size)
        batch = [self.memory[i] for i in indices]
        weights = [1 for i in range(batch_size)]
        return indices, batch, weights


class MultiStepInnerBuffer(InnerBuffer):
    '''
    维护multi-step
    '''

    def __init__(self, capacity=10000, n_steps=1, gamma=0.99):
        self.n_steps = n_steps
        self.capacity = capacity
        if n_steps > self.capacity:
            raise ValueError('n_steps must be equal or less than capacity')
        super().__init__(capacity)
        self.gamma = gamma
        self.multistep_memory = deque(maxlen=capacity)
        self.tot_add = 0
        self.last_episode = -1
        # 记录每个episode的初始位置和结束位置
        self.episode_dict = {}
        self._new_episode_flag = True
        self.cur_window_size = 0
        self.cur_window_value = 0

    def _append(self, data, episode, inner_model=None):
        if self.n_steps == 1:
            self.memory.append(data)
            return
        # 维护相邻的（state1,next_state1） (state2,next_state2) next_state1=state2 这样的大小为n_steps的窗口
        if len(self.memory) == 0:
            self.cur_window_size = 1
            self.cur_window_value = data["reward"]
        else:
            if self.cur_window_size < self.n_steps:
                next_state = self.memory[-1]["next_state"]
                # if set(next_state)==set(data["state"]):
                if (next_state == data["state"]).all():
                    self.cur_window_size = self.cur_window_size + 1
                    self.cur_window_value = data["reward"] * self.gamma ** (
                                self.cur_window_size - 1) + self.cur_window_value
                else:
                    self.cur_window_size = 1
                    self.cur_window_value = data["reward"]
            else:
                next_state = self.memory[-1]["next_state"]
                if set(next_state) == set(data["state"]):
                    self.cur_window_value = (self.cur_window_value - self.memory[-self.cur_window_size][
                        "reward"]) / self.gamma + data["reward"] * self.gamma ** (self.cur_window_size - 1)
                else:
                    self.cur_window_size = 1
                    self.cur_window_value = data["reward"]
        if self.cur_window_size == self.n_steps:
            dic_tmp = self.memory[-self.n_steps + 1].copy()
            dic_tmp["reward"] = self.cur_window_value
            dic_tmp["next_state"] = data["next_state"]
            dic_tmp["done"] = data["done"]
            self.multistep_memory.append(dic_tmp)
        self.memory.append(data)
        self._new_episode_flag = False

    '''
    def _append(self,data,episode):
        self.memory.append(data)
        if self.n_steps == 1:
            return
        episode_len = self.episode_dict[episode][1] - self.episode_dict[episode][0] + 1
        if episode_len > self.n_steps - 1:
            if self._new_episode_flag:
                self._new_episode_flag = False
                result = 0
                for i in range(self.n_steps):
                    dic = self.memory[-(i + 1)]
                    result = dic["reward"] + result * self.gamma
                dic_tmp = self.memory[-self.n_steps].copy()
                dic_tmp["reward"] = result
                self.multistep_memory.append(dic_tmp)
            else:
                last_reward = self.multistep_memory[-1]["reward"]
                last_reward = (last_reward - self.memory[-self.n_steps - 1]["reward"]) / self.gamma
                cur_reward = last_reward + self.gamma ** (self.n_steps - 1) * data["reward"]
                dic_tmp = self.memory[-self.n_steps].copy()
                dic_tmp["reward"] = cur_reward
                dic_tmp["next_state"] = data["next_state"]
                dic_tmp["done"] = data["done"]

                self.multistep_memory.append(dic_tmp)
    '''

    def sample(self, batch_size):
        if self.n_steps == 1:
            indices = random.sample(range(len(self.memory)), batch_size)
            batch = [self.memory[i] for i in indices]
            weights = np.ones(batch_size)

        else:
            indices = random.sample(range(len(self.multistep_memory)), batch_size)
            batch = [self.multistep_memory[i] for i in indices]
            weights = np.ones(batch_size)
        return indices, batch, weights


class ReplayBuffer(metaclass=SerializeMeta):
    def __init__(self,capacity,columns=[]):
        '''

        :param capacity: 缓存区容量
        :param columns:每一条数据有的数据名称，用于约束data
        '''
        self.capacity = capacity
        self.columns = columns
        self.buffer= InnerBuffer(capacity)

    def append(self, data:dict,episode,inner_model=None):
        if sorted(data.keys()) == sorted(self.columns):
            data={key:inner_model.convert_data(key,data[key])for key in data.keys()}
            self._append(data,episode,inner_model)
        else:
            raise Exception("The data can not match the columns.")
    def _append(self,data,episode,inner_model):
        raise NotImplementedError

    def sample(self, batch_size):
        l=len(self)
        if l<batch_size:
            return self._sample(l)
        else:
            return self._sample(batch_size)

    def update(self,indices,current_value,target_value):
        return self.buffer.update(indices, current_value, target_value)
    def __getitem__(self, item):
        return self.buffer[item]
    def __len__(self):
        return len(self.buffer)
    def _sample(self, batch_size):
        raise NotImplementedError
    def can_sample(self, batch_size):
        raise NotImplementedError
class ExperienceReplayBuffer(ReplayBuffer):

    def __init__(self, columns, capacity=10000,gamma=0.99,n_steps=1):
        super().__init__(capacity,columns)
        self.n_steps = n_steps
        self.gamma = gamma
        if n_steps<=0:
            raise ValueError("n_steps must be a positive integer")
        self.buffer = MultiStepInnerBuffer(capacity=capacity,gamma=gamma,n_steps=n_steps)

    def _append(self, data,eposide,inner_model=None):
        self.buffer.append(data,eposide,inner_model)
    def sample(self, batch_size):
        return self._sample(batch_size)
    def can_sample(self, batch_size):
        l=len(self)
        if l<self.n_steps:
            return False
        if self.n_steps==1:
            '''
            直接使用属性不规范
            '''
            return batch_size<=len(self.buffer.memory)
        else:
            return batch_size<=len(self.buffer.multistep_memory)

    def _sample(self, batch_size):
        """随机采样一批经验"""
        indices,batch,weights=self.buffer.sample(batch_size)
        '''
        #转换为{"state":[state0,state1],"reward":[reward0,reward1]...}形式
        result={column:[d[column] for d in batch] for column in self.columns}
        '''
        return indices,batch,weights

