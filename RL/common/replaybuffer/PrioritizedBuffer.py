import numpy as np

from RL.common.replaybuffer import InnerBuffer, ReplayBuffer


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.length = 0

    def __len__(self):
        return self.length

    def get_nbytes(self):
        return self.tree.nbytes + self.data.nbytes  # 8e6 bytes，即 8MB

    # update to the root node
    def _propagate(self, idx):
        parent = (idx - 1) // 2
        idx1 = idx - 1 if idx % 2 == 0 else idx + 1
        self.tree[parent] = self.tree[idx] + self.tree[idx1]
        if parent != 0:
            self._propagate(parent)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        self.tmp_lst.append([idx, s, left, right, self.tree[left], self.tree[right]])
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise IndexError("Index must be an integer")
        if abs(item) > self.capacity:
            raise IndexError("Index exceeded")
        if self.length == self.capacity:
            idx = self.write + item
            return self.data[idx]
        else:
            if item < 0:
                return self.data[self.write + item]
            else:
                return self.data[item]

    # store priority and sample
    def add(self, p, data):
        self.length = min(self.length + 1, self.capacity)
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        # print("last:",self.tree[idx],"change:",change,change.dtype,self.tree[idx].dtype)
        # 不能维护change然后更新父节点，会溢出并且累计
        self.tree[idx] = p
        self._propagate(idx)

    # get priority and sample
    def get(self, s):
        self.tmp_lst = []
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        if not isinstance(self.data[dataIdx], dict):
            for data in self.tmp_lst:
                print(data)
            print("len:", self.length)
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedInnerBuffer(InnerBuffer):
    def __init__(self, capacity,n_steps=1, alpha=0.6, beta=0.4, gamma=0.99, error=0.01):
        super().__init__(capacity)
        self.alpha = alpha  # 优先级调节因子（0=均匀采样，1=完全优先级）
        self.beta = beta  # 重要性采样权重调节因子
        self.gamma = gamma
        self.n_steps = n_steps
        self.tree = SumTree(capacity)
        self.multistep_tree = SumTree(capacity)

        # 保底的error
        self.error = error

    def _append(self, data, episode, inner_model=None):

        target_value = inner_model.get_target_value(data, to_basic_type=True)  # [1,29]
        current_value = inner_model.get_current_value(data, to_basic_type=True)  # [1,29]
        error = self._get_error(current_value, target_value)
        # 存储新经验
        priority = self._get_priority(error)  # [1,29]
        if self.n_steps == 1:
            self.tree.add(priority, data)
            return

        # 维护相邻的（state1,next_state1） (state2,next_state2) next_state1=state2 这样的大小为n_steps的窗口
        if len(self.tree) == 0:
            self.cur_window_size = 1
            self.cur_window_value = data["reward"]

        else:
            if self.cur_window_size < self.n_steps:
                next_state = self.tree[-1]["next_state"]
                if (next_state == data["state"]).all():
                    self.cur_window_size = self.cur_window_size + 1
                    self.cur_window_value = data["reward"] * self.gamma ** (
                                self.cur_window_size - 1) + self.cur_window_value
                else:
                    self.cur_window_size = 1
                    self.cur_window_value = data["reward"]
            else:
                next_state = self.tree[-1]["next_state"]
                if (next_state == data["state"]).all():
                    self.cur_window_value = (self.cur_window_value - self.tree[-self.cur_window_size][
                        "reward"]) / self.gamma + data["reward"] * self.gamma ** (self.cur_window_size - 1)
                else:
                    self.cur_window_size = 1
                    self.cur_window_value = data["reward"]
        if self.cur_window_size == self.n_steps:
            dic_tmp = self.tree[-self.n_steps + 1].copy()
            dic_tmp["reward"] = self.cur_window_value
            dic_tmp["next_state"] = data["next_state"]
            dic_tmp["done"] = data["done"]
            # 有问题
            dic_tmp["length"] = self.n_steps
            current_value = inner_model.get_current_value(data, to_basic_type=True)
            target_value = inner_model.get_target_value(data, to_basic_type=True)
            error = np.abs(target_value - current_value)
            priority = self._get_priority(error)
            self.multistep_tree.add(priority, dic_tmp)

        self.tree.add(priority, data)

    def _get_priority(self, error):
        priorities = (np.abs(error) + self.error) ** self.alpha
        return np.mean(priorities, axis=-1)

    def _get_error(self, current_value, target_value):
        error = np.abs(target_value - current_value)
        return error

    def sample(self, batch_size):
        # 分段采样（解决均匀采样偏差问题）
        if self.n_steps == 1:
            tree = self.tree
        else:
            tree = self.multistep_tree
        segment = tree.total() / batch_size
        indices, transitions, priorities = [], [], []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            idx, priority, data = tree.get(v)
            indices.append(idx)
            priorities.append(priority)
            transitions.append(data)

        # 计算重要性采样权重
        probs = np.array(priorities) / tree.total()
        weights = (len(tree.data) * probs) ** -self.beta
        weights /= weights.max()  # 归一化
        # self.beta=min(1,self.beta+self.beta_increment_per_sampling)
        return indices, np.array(transitions), weights

    def update(self, indices, current_value, target_value):
        td_errors = self._get_error(current_value, target_value)
        priorities = self._get_priority(td_errors)
        if self.n_steps == 1:
            tree = self.tree
        else:
            tree = self.multistep_tree
        for idx, priority in zip(indices, priorities):
            tree.update(idx, priority)

    def __len__(self):
        if self.n_steps == 1:
            return self.tree.length
        else:
            return self.multistep_tree.length
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, columns,capacity=10000, n_steps=1,alpha=0.6, beta=0.4):
        super().__init__(capacity,columns)
        self.alpha = alpha
        self.beta=beta
        self.n_steps = n_steps
        self.buffer = PrioritizedInnerBuffer(capacity=capacity,n_steps=n_steps,alpha=alpha,beta=beta)
    def _append(self, data,eposide,inner_model=None):
        self.buffer.append(data,eposide,inner_model)
        #print("buffer大小：",self.get_nbytes()/1002/1024,"Mb")
    def sample(self, batch_size):
        return self._sample(batch_size)
    def _sample(self, batch_size):
        indices, batch, weights = self.buffer.sample(batch_size)
        return indices, batch, weights
    def can_sample(self, batch_size):
        return batch_size<=len(self.buffer)

class MultiFactorPER(PrioritizedInnerBuffer):
    def __init__(self, capacity, n_steps=1, alpha=0.6, beta=0.4, gamma=0.99, recency_decay=0.99):
        super().__init__(capacity, n_steps, alpha, beta, gamma)
        self.recency_decay = recency_decay
        self.age = np.zeros(capacity, dtype=np.int32)