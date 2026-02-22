import inspect
import io
import json
from collections import deque
import random

import numpy as np
import torch
from torch import nn
from typing import List
from Decorator.Meta import SerializeMeta, SubclassTrackerMeta, Serializer,load_all_modules
from RL.RLBases import ExperienceReplayBuffer, RLInnerModel, ReplayBuffer, RLAgent, InnerBuffer, \
    MultiStepInnerModel



class HRLNodeMeta(SerializeMeta,SubclassTrackerMeta):
    pass

class HRLNode(metaclass=HRLNodeMeta):
    '''
    设置模型的is_primitive属性（如果有），表示当前节点为原子任务节点
    '''
    columns=["state","action","reward","next_state","done","length"]
    def __init__(self, name, model:RLInnerModel,parent=None):
        self.name = name
        self._subtasks = []
        if not set(model.columns).issubset(set(self.columns)):
            raise ValueError("Model update columns must be the subset of the Node update columns")
        self.model = model
        if hasattr(self.model, "is_primitive"):
            self.model.is_primitive = True
        self.parent = parent
        #进入访问时的序号
        self.l_visit_num=-1
        #最近一次访问的序号
        self.r_visit_num=-1
    @property
    def subtasks(self):
        return self._subtasks
    @subtasks.setter
    def subtasks(self,value):
        if not isinstance(value,list):
            raise TypeError("subtasks must be a list")

        for subtask in value:
            if not isinstance(subtask,HRLNode):
                raise TypeError(f"subtask:{subtask} must be an instance of HRLNode")
        self._subtask = value
        if len(self._subtask)!=0:
            if hasattr(self.model, "is_primitive"):
                self.model.is_primitive = False


    def add_subtask(self,subtask):
        if not isinstance(subtask,HRLNode):
            raise TypeError(f"subtask:{subtask} must be an instance of HRLNode")
        self.subtasks.append(subtask)
        if hasattr(self.model,"is_primitive"):
            self.model.is_primitive=False
    def onStepFinished(self,episode,state, next_state, action, reward, done, info):
        self.model.onStepFinished(episode,state,next_state,action,reward,done,info)
    def onEpisodeFinished(self,episode):
        self.model.onEpisodeFinished(episode)
    def append_experience(self,data,episode):
        data1={key:data[key] for key in self.model.columns}
        data1["reward"]=data1["reward"]+self.shaped_reward(data1["next_state"])+self.extrinsic_reward(data1["next_state"])
        self.model.append_memory(data,episode)
    def select_action(self, state):
        return self.model.select_action(state)
    def is_primitive(self):
        return len(self.subtasks) == 0
    def update(self):
        self.model.update()
    def extrinsic_reward(self, state):
        '''
        人为的额外奖励,比如state通过action到达next_state后基于next_state给出奖励/惩罚，目的是加速策略收敛或注入先验知识
        :param state:
        :return:
        '''
        return 0

    def shaped_reward(self, state):
        '''
        奖励塑形,用于策略执行中的每一步，如每走一步若接近终点奖励1，目的是引入额外的奖励信号引导智能体更快学习
        :param state:
        :return:
        '''
        return 0

    def subtask_selection_bonus(self, state):
        '''
        用于引导高层任务（当前节点父亲）,根据状态计算额外奖励从而选择适合的子任务
        :param state:
        :return:
        '''
        return 0

    def pseudo_reward(self, state):
        '''
        子任务结束后的奖励
        :param state:
        :return:
        '''
        return 0
    def is_terminal(self,state):
        return True
    def __eq__(self, other):
        if not self.__class__ == other.__class__:
            return False
        return self.name==other.name

    def __hash__(self):
        return hash((self.__class__, self.name))  # 基于不可变属性生成哈希值


class Tree(metaclass=SerializeMeta):
    def __init__(self, root:HRLNode=None):
        self.root = root
        self.cnt_dic = {}
        self.cnt = 0
        self.node_lst = []
        self.currentNode=root
        self.add_node(self.root,None)
    def add_node(self, node, parent):
        if parent is None:
            self.node_lst.append(node)
            self.cnt = self.cnt + 1
            self.cnt_dic[node] = self.cnt
            return
        if node not in parent.subtasks:
            node.parent = parent
            node.parent.subtasks.append(node)
        if parent not in self.cnt_dic:
            self.node_lst.append(parent)
            self.cnt = self.cnt + 1
            self.cnt_dic[parent] = self.cnt
        if node not in self.cnt_dic:
            self.node_lst.append(node)
            self.cnt = self.cnt + 1
            self.cnt_dic[node] = self.cnt

    def load_from_root(self, root):
        self.root = root
        self.currentNode=root
        self.cnt = 0

        def load(node):
            for child in node.subtasks:
                load(child)
                self.add_node(child, node)

        load(root)
    def load_from_json(self, json_file):
        with open(json_file, 'r') as f:
            dic = json.load(f)
        obj_dic={}
        nodes=dic["nodes"]
        module_paths=dic["module_paths"]
        for module_path in module_paths:
            load_all_modules(module_path)
        for key in nodes.keys():
            tmp=nodes[key]
            node_type=tmp["node_type"]
            params = tmp["params"]
            subtasks = tmp["subtasks"]
            cls=HRLNode.get_class(node_type)

            obj=cls(**params)
            obj_dic[key]=obj
            obj.subtasks=subtasks

        self.root=obj_dic[dic["root"]]
        self.cnt=0
        self.cnt_dic={}
        self.node_lst=[]
        self.add_node(self.root,None)
        for key in obj_dic.keys():
            obj_dic[key].subtasks=[obj_dic[subtask] for subtask in obj_dic[key].subtasks]
            for subtask in obj_dic[key].subtasks:
                self.add_node(subtask, obj_dic[key])

    def config_to_dic(self):
        result={"root":self.root.name,"nodes":{},"module_paths":[]}
        dic=result["nodes"]
        def dfs(node):
            dic[node.name]={}
            now=dic[node.name]
            now["node_type"]=node.__class__.__name__
            # 获取 __init__ 方法的参数信息
            init_signature = inspect.signature(node.__class__.__init__)
            parameters = list(init_signature.parameters.values())
            # 过滤掉 self 参数
            constructor_params = [p for p in parameters if p.name != 'self']
            now["params"]={p.name:getattr(node,p.name) for p in constructor_params}
            now["subtasks"]=[child.name for child in node.subtasks]
            for child in node.subtasks:
                dfs(child)


        dfs(self.root)

        return result

    def get_tree_info(self):
        '''

        :return:节点树嵌套信息
        '''
        info = {}

        def dfs(node, now):
            for child in node.subtasks:
                k = f"{child.name}({child.__class__})"
                now[k] = {}
                dfs(child, now[k])

        k = f"{self.root.name}({self.root.__class__})"
        info[k] = {}
        dfs(self.root, info[k])

        return info

    def __len__(self):
        return self.cnt

    def save_tree(self, filename):
        dic = self.to_dict()
        with open(filename, 'w') as f:
            json.dump(dic, f)

    @classmethod
    def load_tree(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        obj = cls.from_dict(data)

        return obj

class HRLInnerBuffer(InnerBuffer):
    '''
       维护任意steps的奖励查询
       reward=[r1,r2,r3,r4,r5]*[1,gamma,gamma^2,gamma^3,gamma^4]点乘
    '''

    def __init__(self, capacity=10000,gamma=0.99):
        super().__init__(capacity=capacity)
        self.gamma=gamma
    def query_reward(self,l,r):
        now=1
        reward=0
        for i in range(l-1,r):
            reward+=now*self.memory[i]["reward"]
            now=now*self.gamma
        return reward


class HRLReplayBuffer(ReplayBuffer):

    def __init__(self, columns, capacity=10000, gamma=0.99):
        super().__init__(capacity=capacity,columns=columns)
        self.gamma = gamma
        self.buffer = HRLInnerBuffer(capacity=capacity, gamma=gamma)
    def query_reward(self,l,r):
        return self.buffer.query_reward(l,r)
    def _append(self,data,episode,inner_model=None):
        self.buffer.append(data,episode)

    def can_sample(self, batch_size):
        return len(self.buffer) >= batch_size
    def _sample(self, batch_size):
        return self.buffer.sample(batch_size)
class HRLInnerModel(RLInnerModel):
    '''
    整颗树抽象成一个模型(和整体设计保持一致)
    '''

    def __init__(self,memory_capacity=10000,gamma=0.99,exploration_strategy=None,tree=None,batch_size=32):
        super().__init__(memory_capacity=memory_capacity,gamma=gamma,batch_size=batch_size,exploration_strategy=exploration_strategy)
        self.tree = tree
        self.gamma=gamma
        #记录每一步要更新的节点
        self.step_update_list=[]
    def set_default_memory(self):
        self.memory = HRLReplayBuffer(capacity=self.memory_capacity, gamma=self.gamma,
                                      columns=self.__class__.columns)

    def _select_action(self,state):
        while not self.tree.currentNode.is_primitive():
            #要修改
            self.tree.currentNode=self.tree.currentNode.select_action(state)
            self.tree.currentNode.l_visit_num=self.steps_done
        if self.tree.currentNode.l_visit_num==-1 or self.tree.currentNode.r_visit_num==-1:
            self.tree.currentNode.l_visit_num=self.steps_done
            self.tree.currentNode.r_visit_num=self.steps_done
        else:
            self.tree.currentNode.r_visit_num=self.steps_done

        return self.tree.currentNode.select_action(state)
    def onStepFinished(self,episode,state,next_state,action,reward,done,info):
        self.step_update_list=[]
        while done or self.tree.currentNode.is_terminal(next_state):
            len1 = self.steps_done - self.tree.currentNode.l_visit_num + 1
            r = len(self.memory) - 1
            l=r-len1+1

            reward1=self.memory.query_reward(l,r)
            state1 = self.memory[-len1]["state"]
            #assert state1==state
            self._store_hierarchical_exp(self.tree.currentNode,episode,state1,action,reward1,next_state,done,len1)
            self.tree.currentNode.l_visit_num = self.tree.currentNode.r_visit_num = -1
            self.tree.currentNode.onStepFinished(episode,state, next_state, action, reward, done, info)
            #self.tree.currentNode.update()
            self.step_update_list.append(self.tree.currentNode)
            if self.tree.currentNode.parent:
                self.tree.currentNode.parent.r_visit_num=self.tree.currentNode.r_visit_num
                self.tree.currentNode=self.tree.currentNode.parent
            else:
                break
        self.tree.currentNode.onStepFinished(episode,state, next_state, action, reward, done,info)
        #self.tree.currentNode.update()
        self.step_update_list.append(self.tree.currentNode)
    def onEpisodeFinished(self,episode):
        #self.tree.currentNode=self.tree.root
        def dfs(node):
            self.tree.currentNode.onEpisodeFinished(episode)
            for subtask in node.subtasks:
                dfs(subtask)
        dfs(self.tree.root)

    def _store_hierarchical_exp(self, node,episode,state, action, reward, next_state,done,length):
        """分层存储经验到对应节点"""
        data={"state":state, "action":action, "reward":reward, "next_state":next_state, "done":done,"length":length}
        node.append_experience(data,episode)

    def get_target_value(self,data,to_basic_type):
        pass
    def get_current_value(self,data,to_basic_type):
        pass

    def get_action_scores(self,state):
        pass
    def _update(self,indices,batch_data,weights):
        for node in self.step_update_list:
            node.update()

class HRLNodeInnerModelDecorator:

    '''
    增加is_primitive属性
    '''
    @classmethod
    def Decorator(cls_decorator,cls):
        class NewClass(cls):
            def __init__(self,*args,is_primitive=None,**kwargs):
                super().__init__(*args,**kwargs)
                self.is_primitive = is_primitive

        return NewClass

    def __new__(cls_decorator, cls):
        return cls_decorator.Decorator(cls)
'''
class HRLNodeInnerModel(MultiStepInnerModel):
    
    如果是叶子节点的原子任务则n_steps参数生效，表示模型采用多步的采样，在更新模型时memory中的length项无用
    当错误的设置高层任务为原子任务，高层任务记忆中的每一项的length不全为1，而memory内部维护的多步采样默认是
    连续的两个状态（length=1)而不是两个跳跃若干步后的状态，这显然会出错

     如果不是叶子节点则n_steps参数失效，模型会使用length来更新模型
    
    columns = ["state", "action", "reward", "next_state", "done", "length"]
   

    def __init__(self,memory=None,is_primitive=False,n_steps=1, batch_size=32, memory_capacity=10000, gamma=0.99,
                 exploration_strategy=None):
        super().__init__(memory=memory,n_steps=n_steps, batch_size=batch_size, gamma=gamma, memory_capacity=memory_capacity,
                         exploration_strategy=exploration_strategy)
        self.is_primitive =is_primitive
'''
