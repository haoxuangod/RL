import math

import numpy as np

from RL.HierarchicalRL.MaxQ import MaxQNode


class TestTask(MaxQNode):
    def is_terminal(self,state):
        theta = abs(state[2])  # 杆子角度绝对值
        # 失败条件：角度超过阈值
        if theta > math.pi/12:
            return True
        '''
        n=5
        # 成功条件（可选）：next_state 连续稳定
        if len(self.memory) >= n:
            last_n = [self.memory[-i][3] for i in range(n)]
            if all([abs(s[2]) < 0.0174 for s in last_n]):
                return True
        return False
        '''
    def shaped_reward(self, state):
        theta = abs(state[2])
        return np.exp(8 * theta)  # 角度越小奖励越高