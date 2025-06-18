import io
import json
import time
import random

import pygame
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch import nn
from Decorator.Meta import *
import gym
import torch
import numpy as np
from collections import deque

from RL.HierarchicalRL.HRLBases import HRLNode, Tree
from Model.Model import IterationModel, ModelData

class MaxQNode(HRLNode):

    def select_action(self, state, epsilon):
        if not self.is_primitive():
            not_terminated = [0 if task.is_terminal(state) else 1 for task in self.subtasks]
        else:
            not_terminated = [1] * self.action_dim

        if np.random.rand() < epsilon:
            # 获取非零元素索引
            nonzero_indices = np.nonzero(not_terminated)[0]
            assert len(nonzero_indices) != 0
            # 随机选择一个非零位置
            random_index = np.random.choice(nonzero_indices)
            # return np.random.randint(self.action_dim)
            return random_index
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.q_net(state_tensor)
                if not self.is_primitive():
                    subtask_selection_bonuses = [task.subtask_selection_bonus(state) for task in self.subtasks]
                    subtask_selection_bonuses = torch.FloatTensor(subtask_selection_bonuses)
                    q_values = q_values + subtask_selection_bonuses

            not_terminated = torch.FloatTensor(not_terminated)
            return torch.argmax(q_values * not_terminated).item()
    def update(self, batch_size, gamma):
        batch_data=self.memory.sample(batch_size)
        states = batch_data["state"]
        actions = batch_data["action"]
        rewards=batch_data["reward"]
        next_states=batch_data["next_state"]
        rates = batch_data["rate"]
        dones=batch_data["done"]
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        rates = torch.FloatTensor(rates)
        dones = torch.FloatTensor(dones)
        # MAXQ核心计算
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():

            target_q = rewards.unsqueeze(1) + gamma*rates.unsqueeze(1) * (1 - dones.unsqueeze(1)) * \
                        self.target_net(next_states).max(1)[0].unsqueeze(1)

                # 计算损失
        loss = torch.nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class MaxQTree(Tree):
    pass


class HierarchicalMAXQAgent(IterationModel):
    def __init__(self,  name, env,tree,render=True,max_iter=1000,
                 print_interval=10,
                 save_interval=100,
                 sync_target_interval=10,
                 directory='', save_name='',gamma=0.99, batch_size=32):
        self.tree=tree
        self.gamma = gamma
        self.batch_size = batch_size
        self.env=env
        self.epsilon = 1.0
        self.render=render
        self.sync_target_interval=sync_target_interval
        data=EnvModelData(name=env.__class__.__name__,filepath=None)
        super().__init__(name,data,max_iter,print_interval,save_interval,directory,save_name)
        self.interval_callbacks.append((self.sync_target_interval,self.sync_target_network))
        self.game_window=GameWindow(screen_width=800, screen_height=600,rewards_window_length=200)


    def load_state(self):
        super().load_state()
        self.tree=from_dict(self.state.parameters["tree"])

    def update_state(self):
        """
        更新模型状态，保存当前优化器的状态到 ModelState。
        """
        super().update_state()
        try:
            state = {
                "tree":self.tree.to_dict()
            }

            # print("update_state:",state)
            # 更新 ModelState 的参数
            self.state.update_parameters(state)
        except Exception as e:
            print(f"Iteration Model:更新优化器状态时出错: {e}")

        #HierarchicalMAXQAgent._reset()
    def train_iteration(self,iteration):
        return self.execute_episode(self.env,self.render)
    def execute_episode(self, env,render=True):
        state,_ = env.reset()
        global done
        done = False
        def maxq_learn(current_node,state):
            global done
            total_reward = 0
            reward_train=0
            rate=1
            while not current_node.is_terminal(state) and not done:
                action = current_node.select_action(state, self.epsilon)
                if current_node.is_primitive():
                    # 执行原始动作
                    '''
                    terminated 表示环境自然结束（如任务成功/失败）
                    示例：
                        机器人摔倒（失败）
                        平衡杆完全倒下（任务终止）
                    truncated:表示人为截断（非MDP定义的终止条件）
                    触发场景：
                        达到最大步数限制（如max_episode_steps）
                        中途强制结束（如机器人掉下悬崖但未到终止状态）
                    '''
                    next_state, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    shaped_reward=current_node.shaped_reward(next_state)
                    reward_train += rate * (reward+shaped_reward)
                    done = terminated or truncated  # 合并终止标志
                    rate *= self.gamma

                else:
                    subtask=current_node.subtasks[action]
                    next_state,reward,sub_rate=maxq_learn(subtask,state)
                    total_reward+=reward
                    shaped_reward = current_node.shaped_reward(next_state)
                    subtask_pseudo_reward=subtask.pseudo_reward(next_state)
                    reward_train+=rate*(reward+shaped_reward+subtask_pseudo_reward)
                    rate*=sub_rate

                # 存储分层经验
                self._store_hierarchical_exp(current_node, state, action, reward_train, next_state,rate, done)
                # 更新网络参数
                self._hierarchical_update(current_node)
                state = next_state


            return state,total_reward,rate
        state,total_reward,rate=maxq_learn(self.tree.root,state)
        reward=0
        if render:
            self.game_window.render_window(env,reward,total_reward,self.history)
        self.epsilon *= 0.995  # 探索率衰减

        print(f"回合结束，最终得分: {total_reward}")
        return total_reward


    def sync_target_network(self):
        def sync_target(node):
            for child in node.subtasks:
                if not child.is_primitive:
                    sync_target(child)
            node.target_net.load_state_dict(node.q_net.state_dict())

        sync_target(self.tree.root)
    def _store_hierarchical_exp(self, node,state, action, reward, next_state,rate,done):
        """分层存储经验到对应节点"""
        data={"state":state, "action":action, "reward":reward, "next_state":next_state, "rate":rate, "done":done}
        node.memory.add(data)

    def _hierarchical_update(self,node):
        if len(node.memory) >= self.batch_size:
            node.update(self.batch_size, self.gamma)

