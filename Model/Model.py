import asyncio
import copy
import os
import json
import time
import traceback

import pandas as pd
import requests
from dask.distributed import LocalCluster, Queue, get_client
import dill as pickle
from distributed.comm.inproc import QueueEmpty

'''
2024-11-18
1.各个操作的线程安全/异常处理
2.询问单个模型运行的情况
3.Model加上状态模式
4.完善运行状态的询问
5.想办法精简模型的大小    
6.抽象一个迭代运行的模型子类，NPSO继承它 √
7.模型的读取 √
8.传文件时损坏 √
9.运行结束后指标统计 
'''


def get_time():
    return time.time()*1000
class State:
    def __init__(self):
        '''
        模型的state 是 running还是finished
        '''
        self.state=""
        self.time=get_time()

    def load(self, filename):
        pass
    def __lt__(self, other):
        return self.time<other.time
    def __gt__(self, other):
        return self.time> other.time
    def set_state(self, state):
        self.state=state
    @staticmethod
    def create_state(filename):
        '''
        从文件中读取一个state对象
        :param filename:
        :return:
        '''
    def get_data(self):
        raise NotImplementedError("get_data is not implemented")
    def save(self, filename):
        state_data = self.get_data()
        save_path=filename
        save_path_tmp=filename+".tmp"
        try:
            # 定义保存路径，按时间戳或唯一标识符命名
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path_tmp, 'wb') as f:
                pickle.dump(state_data, f)
            os.replace(save_path_tmp, save_path)
            print(f"已保存状态文件到 {save_path}")
        except Exception as e:
            if os.path.exists(save_path_tmp):
                os.remove(save_path_tmp)  # 清理临时文件
            print(f"保存状态文件到本地时出错: {e}")



class ModelState(State):
    def __init__(self, model_name,data, parameters=None):
        super().__init__()
        self.model_name = model_name
        self.data=data
        self.parameters = parameters if parameters is not None else {}
    def set_parameter(self, key,value):
        if key in self.parameters:
            self.parameters[key] = value
            self.time=get_time()
    def update_parameters(self, dic):
        self.parameters.update(dic)
        self.time=get_time()

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                state_data = pickle.load(f)
                self.model_name = state_data['model_name']
                self.parameters = state_data.get('parameters', {})
                self.data=state_data["data"]
                self.time=state_data['time']
                self.state=state_data['state']
        else:
            print(f"模型状态文件 {filename} 不存在")
    @staticmethod
    def create_state(filename):
        state=ModelState("aaa",None)
        state.load(filename)
        return state
    def get_data(self):
        state_data = {
            'model_name': self.model_name,
            'parameters': self.parameters,
            "data":self.data,
            "state":self.state,
            "time":self.time
        }
        return state_data


class IterationModelState(ModelState):
    def __init__(self, model_name, data,current_iteration,max_iterations,parameters=None):
        super().__init__(model_name, data,parameters)
        self.model_name = model_name
        self.data = data
        self.parameters = parameters if parameters is not None else {}
        self.parameters.update({"current_iteration":current_iteration,"max_iterations":max_iterations})


    @staticmethod
    def create_state(filename):
        state = IterationModelState("aaa", None,-1,-1)
        state.load(filename)
        return state

    def __lt__(self, other):
        return self.parameters['current_iteration']<other.parameters['current_iteration']

    def __gt__(self, other):
        return self.parameters['current_iteration']>other.parameters['current_iteration']


class TrainState(State):
    def __init__(self, model_name, data, current_run=0, num_runs=0):
        super().__init__()
        self.model_name = model_name
        self.data = data
        self.current_run = current_run
        self.num_runs = num_runs
        self.status={} #{"save_name":{"result":xxx,"state":state}

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                state_data = pickle.load(f)
                self.model_name = state_data['model_name']
                self.data = state_data['data']
                self.current_run = state_data['current_run']
                self.num_runs = state_data['num_runs']
                self.state=state_data['state']
                self.status = state_data.get('status', [])
        else:
            print(f"状态文件 {filename} 不存在")

    def get_data(self):
        state_data = {
            'model_name': self.model_name,
            'data': self.data,
            'current_run': self.current_run,
            'num_runs': self.num_runs,
            'status': self.status,
            'state':self.state
        }
        return state_data

class DataProcessor:
    def __init__(self,process_func,preProcessor=None):
        self.process_func = process_func
        self.preProcessor =preProcessor
    def process(self,data):
        if self.preProcessor is not None:
            data = self.preProcessor.process(data)
        return self._process(data)

    def _process(self,data):
        '''
        实际需要重写的函数
        '''
        return self.process_func(data)
    def __call__(self,data):
        return self.process(data)
class ModelData:
    def __init__(self, name, filepath, url=None, preprocesser=None):
        """
        初始化 ModelData 类。

        参数:
        - name: 数据集名称。
        - filepath: 数据文件的路径。
        - url: 数据集的下载URL（如果需要下载）。
        - preprocessing: 数据预处理函数或方法。
        - preprocessing_kwargs: 数据预处理函数给定的额外参数（字典形式）
        """
        self.name = name
        self.filepath = filepath
        self.url = url
        self.preprocesser = preprocesser  # 可选的预处理函数

    def download_data(self):
        """
        下载数据集并保存到指定的文件路径。
        """
        if self.url is None:
            print(f"数据集 '{self.name}' 没有提供下载URL。")
            return False

        try:
            print(f"开始下载数据集 '{self.name}' 从 {self.url}...")
            response = requests.get(self.url)
            response.raise_for_status()  # 检查请求是否成功

            with open(self.filepath, 'wb') as f:
                f.write(response.content)
            print(f"数据集 '{self.name}' 下载并保存到 {self.filepath} 成功。")
            return True
        except Exception as e:
            print(f"下载数据集 '{self.name}' 时出错: {e}")
            return False

    def load_data(self):
        """
        加载数据集。

        返回:
        - pandas.DataFrame: 加载的数据集。
        """
        if not os.path.exists(self.filepath):
            print(f"数据文件 {self.filepath} 不存在。")
            if self.url:
                success = self.download_data()
                if not success:
                    return None
            else:
                return None

        # 获取文件扩展名并转换为小写
        _, ext = os.path.splitext(self.filepath)
        ext = ext.lower()

        try:
            if ext == '.csv':
                data = pd.read_csv(self.filepath)
            elif ext == '.json':
                data = pd.read_json(self.filepath)
            elif ext in ['.xls', '.xlsx']:
                data = pd.read_excel(self.filepath)
            elif ext == '.parquet':
                data = pd.read_parquet(self.filepath)
            elif ext == '.feather':
                data = pd.read_feather(self.filepath)
            elif ext in ['.pickle', '.pkl']:
                data = pd.read_pickle(self.filepath)
            else:
                print(f"不支持的文件格式: {ext}")
                return None

            print(f"数据集 '{self.name}' 加载成功，包含 {data.shape[0]} 行和 {data.shape[1]} 列。")

            if self.preprocesser:
                data = self.preprocesser.process(data)
                print(f"数据集 '{self.name}' 已完成预处理。")

            return data

        except Exception as e:
            print(f"加载数据文件 {self.filepath} 时出错: {e}")
            return None

    def __str__(self):
        return f"ModelData(name='{self.name}', filepath='{self.filepath}')"
class Model:
    def __init__(self, name, data:ModelData,directory='',save_name=''):
        self.name = name
        self.data=data
        self.state = ModelState(name,self.data)
        self.set_directory(directory)
        '''
        比如第i次运行的保存状态名称
        '''
        self.save_name =save_name
    def set_directory(self, directory):
        self.directory = directory
        self.ensure_directory()
    def ensure_directory(self):
        if self.directory and not os.path.exists(self.directory):
            os.makedirs(self.directory)
    def get_state_filename(self,save_name=None):
        '''

        :param save_name: 目的是为了一个对象但是不同的save_name能够获取到filename
        :return:
        '''
        if save_name is None:
            save_name=self.save_name
        filename = f"{self.name}_{self.data.name}_model_state_{save_name}.pkl"
        return os.path.join(self.directory, filename) if self.directory else filename
    def _train(self,*args,**kwargs):
        # 在这里添加实际的训练逻辑
        print(f"正在训练模型 {self.name}，使用数据 {self.data}")
        # 模拟训练结果，并更新模型参数
        result = {"accuracy": 0.9, "loss": 0.1}
        # 假设模型参数在训练过程中更新
        self.state.set_parameter('last_accuracy',result['accuracy'])
        self.state.set_parameter('last_loss', result['loss'])
        self.save()
        return result
    def train(self,*args,**kwargs):
        """
        基础的训练方法，应在子类中重写。
        """

        '''
        raise NotImplementedError("Subclasses should implement this method.")
        '''
        self.state.set_state("running")
        result=self._train(*args,**kwargs)
        self.finish()
        return result
    def finish(self):
        '''
        训练结束时调用的函数
        :return:
        '''
        self._finish()
    def _finish(self):
        self.state.set_state("finished")
    def load(self):
        '''
        从状态文件中加载状态
        :return:
        '''
        filename = self.get_state_filename()
        self.state.load(filename)
        self.load_state()
    def load_from_file(self,file):
        if not os.path.exists(file):
            raise FileNotFoundError(f"{file} does not exist")
        if not os.path.isfile(file):
            raise ValueError(f"{file} is not a file")
        self.state.load(file)
        self.load_state()
    def load_state(self):
        '''
        从state中加载对象的参数
        :return:
        '''
    def set_state(self,state:ModelState):
        '''
        设置状态并且更新对象内的属性
        :param state:
        :return:
        '''
        self.state=state

    def update_state(self):
        '''
        子类实现,用于更新状态对象
        :return:
        '''
    def save(self):
        filename = self.get_state_filename()
        self.state.save(filename)
    def set_save_name(self, save_name):
        self.save_name=save_name
    def reset(self):
        '''
        重置运行结果
        :return:
        '''
        self._reset(run_super=True)
    def _reset(self,run_super=False):
        '''
        为了解决在__init__中父类多次调用_reset方法，添加了run_super参数
        重新时内部调用update_state得调用自己类的不能self，否则会调用为子类的
        :param run_super:
        :return:
        '''
        self.state=ModelState(self.name,self.data)
    def get_progress(self):
        return self.state.get_data()

    def __str__(self):
        return self.name
class IterationModel(Model):
    '''
    运用于需要多次迭代的模型，自动记录运行时间，自动保存进度，隔一定迭代次数自动输出
    '''
    def __init__(self, name, data: ModelData, max_iter=1000,
                 print_interval=1000,
                 save_interval=10000,
                 directory='', save_name=''):
        super().__init__(name, data, directory, save_name)
        self.max_iter = max_iter
        self.print_interval = print_interval
        self.save_interval =save_interval
        #存储每隔几次迭代运行一次函数[(interval,callback_func),]
        self.interval_callbacks=[]
        #不能带self，因为会调用子类的_reset
        #self._reset()
        IterationModel._reset(self)

    def set_state(self,state:IterationModelState):
        self.state=state
    def load_state(self):
        self.current_iteration = self.state.parameters["current_iteration"]
        self.history=self.state.parameters["history"]
        self.max_iter=self.state.parameters["max_iterations"]
        self.total_time=self.state.parameters["total_time"]
    def update_state(self):
        """
        更新模型状态，保存当前优化器的状态到 ModelState。
        """
        try:
            state = {
                'history': self.history.copy(),
                'current_iteration': self.current_iteration,
                'max_iterations': self.max_iter,
                "total_time": self.total_time
            }
            #print("update_state:",state)
            # 更新 ModelState 的参数
            self.state.update_parameters(state)
        except Exception as e:
            print(f"Iteration Model:更新优化器状态时出错: {e}")
    def _reset(self,run_super=False):

        if run_super:
            super()._reset(run_super=run_super)
        self.history = []
        self.current_iteration = 0  # 当前迭代次数
        self.total_time = 0.0  # 总运行时间
        self.total_iterations = 0  # 已执行的迭代次数
        self.state=IterationModelState(self.name, self.data, 0,self.max_iter)
        IterationModel.update_state(self)
    def add_interval_callback(self,callback):
        #待进行判断
        self.interval_callbacks.append(callback)
    def train_iteration(self,iteration,*args,**kwargs):
        """
        一次迭代干的事情
        :param iteration: 当前迭代数
        :return:
        """
    def add_history(self,data):
        self.history.append(data)
    def _train(self,*args,**kwargs):
        try:
            recent_total_time = 0.0  # 用于记录最近 print_interval 次迭代的总运行时间
            recent_iterations = 0  # 用于记录最近 print_interval 次迭代的次数
            for t in range(self.current_iteration, self.max_iter):
                start_time = time.time()  # 记录迭代开始时间
                self.current_iteration = t  # 更新当前迭代次数
                result=self.train_iteration(t,*args,**kwargs)
                end_time = time.time()  # 记录迭代结束时间
                iteration_time = end_time - start_time  # 计算迭代运行时间
                self.total_time += iteration_time  # 累加总运行时间
                self.total_iterations += 1  # 增加已执行的迭代次数
                recent_total_time += iteration_time  # 累加最近 print_interval 次迭代的总运行时间
                recent_iterations += 1  # 增加最近 print_interval 次迭代的次数
                for interval,callback in self.interval_callbacks:
                    if (t+1)%interval==0:
                        callback()

                # 判断是否需要打印和保存状态
                if (t + 1) % self.print_interval == 0 or (t + 1) == self.max_iter:
                    print(
                        f"Model:{self.name}_{self.save_name} 迭代 {t + 1}/{self.max_iter}"
                        f" 最近 {recent_iterations} 次迭代总时间: {recent_total_time:.6f} 秒 运行结果：{result}")
                    # 重置最近的累加变量
                    recent_total_time = 0.0
                    recent_iterations = 0
                    self.current_iteration = t + 1
                # 更新并保存当前状态
                if (t + 1) % self.save_interval == 0:
                    #print("self.current_iteration:",self.current_iteration," save_state iteration_before:", self.state.parameters['current_iteration'])
                    self.update_state()
                    self.save()
                    #print("save_state iteration_after:",self.state.parameters['current_iteration'])
                self.add_history(result)
            return self.get_train_result()
        except Exception as e:
            error_info = traceback.format_exc()
            print(f"Model:{self.name}_{self.save_name} 优化过程中出错\nERRORINFO:{error_info}")
            return self.get_train_result(is_error=True)
        #raise NotImplementedError("_train is not implemented")
    def get_train_result(self,is_error=False):
        '''
        :param is_error: 训练过程中错误返回什么
        :return:
        '''
        #raise NotImplementedError("get_train_result is not implemented")
        return {"result":None,"is_error":is_error}
    def get_average_iteration_time(self):
        """
        获取所有迭代的平均运行时间。

        返回:
            float: 平均迭代时间（秒）。
        """
        if self.total_iterations == 0:
            return 0.0
        return self.total_time / self.total_iterations
from dask import delayed
import dask.array as da
from dask.distributed import Client


class DaskModel:
    '''
    一个代理模型用于将一个可序列化的Model对象添加自动上传状态功能
    '''
    def __init__(self,model:Model,state_queue_name:str,scheduler_address:str, upload_interval=60,):
        """
        初始化 DaskModel。

        参数:
        - model: 代理模型对象。
        - state_queue_name: Dask 的共享队列名称。
        - scheduler_address: Dask Scheduler 地址。
        - upload_interval: 状态文件上传的时间间隔（秒）。
        -
        """
        self.scheduler_address = scheduler_address
        self.model = model
        self.upload_interval = upload_interval  # 上传间隔
        self.state_queue_name = state_queue_name
        self.upload_thread = threading.Thread(target=self.periodic_upload, daemon=True)
        self.stop_event = threading.Event()
        self.client = Client(scheduler_address)
    def train(self):
        self.start_upload_thread()
        result=self.model.train()
        self.stop_upload_thread()
        return result
    def start_upload_thread(self):
        """
        启动上传线程。
        """
        if self.state_queue_name:
            self.upload_thread.start()
            print(f"模型 {self.model.name} 的上传线程已启动。")
        else:
            print(f"模型 {self.model.name} 没有指定状态队列，无法启动上传线程。")

    def stop_upload_thread(self):
        """
        停止上传线程。
        """
        self.stop_event.set()
        self.upload_thread.join()
        print(f"模型 {self.model.name} 的上传线程已停止。")

    def upload_state(self):
        """
        将当前模型的状态以 JSON 格式上传到共享队列。
        """
        try:

            state_queue=Queue(name=self.state_queue_name)
            state = self.model.state
            state_queue.put({"save_name":self.model.save_name,"state":state})
            print(f"模型 {self.model.name} save_name:{self.model.save_name} 已上传状态到队列 迭代次数：",state.parameters['current_iteration'])
        except Exception as e:
            print(f"模型 {self.model.name} save_name:{self.model.save_name} 上传状态时出错: {e}")

    def periodic_upload(self):
        """
        定期上传模型状态。
        """
        while not self.stop_event.is_set():
            time.sleep(self.upload_interval)
            self.upload_state()




class TrainingRun:
    def __init__(self, model, data, num_runs, plan_name, directory=''):
        self.model = model
        '''
        data后续需要抽象成为类
        '''
        self.data = data
        self.num_runs = num_runs
        self.plan_name = plan_name
        self.directory = directory  # 在构造函数中设置目录
        self.ensure_directory()

        self.state = TrainState(model.name, data, num_runs=num_runs)
        self.filename = f"{self.plan_name}_{self.model.name}_{self.data.name}_progress.pkl"
        self.progress_file = os.path.join(self.directory, self.filename) if self.directory else self.filename
        # 设置模型的目录，并加载模型状态
        self.model.set_directory(self.directory)
        self.model.data=data
    def ensure_directory(self):
        if self.directory and not os.path.exists(self.directory):
            os.makedirs(self.directory)
    def run(self):

        # 加载训练进度
        self.load_progress()

        for i in range(self.state.current_run, self.num_runs):
            print(f"model:{self.model.name} process:{i + 1}/{self.num_runs} start")
            save_name=f"{i+1}"
            self.model.set_save_name(save_name)
            self.model.reset()
            self.model.load()
            result = self.model.train()
            if result["is_error"]:
                print(f"model:{self.model.name} process:{i} 出错，退出运行 ")
                break
            '''
            error:模型训练出错应该有处理
            '''
            self.state.status[save_name]={"result":result,"state":copy.deepcopy(self.model.state)}
            self.state.current_run += 1
            self.record_progress()
            # 保存模型状态
            self.model.save()
            print(f"model:{self.model} process:{i+1}/{self.num_runs} finished")
            # 可以在这里添加中断逻辑

        # 运行结束后调用总结方法
        self.summarize_results()
        self.record_progress()

    def record_progress(self):
        self.state.save(self.progress_file)
    def load_progress(self):
        if os.path.exists(self.progress_file):
            self.state.load(self.progress_file)
            self.model.set_save_name(f"{self.state.current_run}")
            self.model.load()

        else:
            self.state.current_run = 0
            self.state.results = []

    def get_progress(self):
        return {
            'model_name': self.model.name,
            'data': self.data,
            'current_run': self.state.current_run,
            'num_runs': self.state.num_runs,
            'model_progress':self.model.get_progress()
        }

    def get_results(self):
        return self.state.results

    def summarize_results(self):
        # 默认不执行任何操作，但可以在子类中重写
        pass
    def set_directory(self, directory):
        self.directory = directory
        self.progress_file = os.path.join(self.directory, self.filename) if self.directory else self.filename
        self.model.set_directory(self.directory)


def execute_training_run(model,model_run_directory,state_queue_name,scheduler_address,upload_interval=60):
        '''
        执行单个训练运行的逻辑，不在DaskTrainingRun内是为了不序列化DaskTrainingRun对象
        '''
        model.set_directory(model_run_directory)
        model_instance = DaskModel(model=model,state_queue_name=state_queue_name,scheduler_address=scheduler_address,upload_interval=upload_interval)
        result = model_instance.train()

        #停止模型运行
        #print("result:",type(result["gbest"][0]),type(result["gbest_value"]),type(model_instance.model.save_name))
        #return {"result": None, "state":model_instance.model.state.get_data(), "save_name": model_instance.model.save_name}
        #print("run:", {"result":result,"state":model_instance.model.state.get_data(),"save_name":model_instance.model.save_name})
        return {"result":result,"state":model_instance.model.state,"save_name":model_instance.model.save_name}


class DaskTrainingRun(TrainingRun):
    '''
    将多次运行提交到 Dask 中，每个运行使用独立的模型实例
    '''

    def __init__(self, model, data, num_runs, plan_name, directory='',monitor_sleep_interval=1,monitor_queue_get_timeout=5,model_run_directory=None,scheduler_address=None, upload_interval=60,
                 state_queue_name=None):
        '''

        :param model:
        :param data:
        :param num_runs:
        :param plan_name:
        :param directory:
        :param monitor_sleep_interval:queue_get未获取信息休息多久再次获取
        :param monitor_queue_get_timeout:queue_get等待多少时间算超时
        :param model_run_directory: 模型分布式运行节点的工作路径
        :param scheduler_address:
        :param upload_interval:
        :param state_queue_name:
        '''
        super().__init__(model, data, num_runs, plan_name, directory)
        self.monitor_sleep_interval=monitor_sleep_interval
        self.monitor_queue_get_timeout=monitor_queue_get_timeout
        self.model_run_directory = directory if model_run_directory is None else model_run_directory
        self.scheduler_address = scheduler_address or getattr(self.model, 'scheduler_address', None)
        self.lock = threading.Lock()  # 用于保护 self.state
        self.upload_interval = upload_interval  # 上传间隔

        if state_queue_name is None:
            self.state_queue_name =f"{plan_name}_{model.name}_{data}_queue"
        else:
            self.state_queue_name = state_queue_name  # 共享队列实例
        self.monitor_thread = None
        self.stop_event = threading.Event()

    def load_progress(self):
        if os.path.exists(self.progress_file):
            self.state.load(self.progress_file)
            print("load_progress:",self.state.status)
        else:

            self.state.current_run = 0
            self.state.results = []


    def run(self):

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self.monitor_progress, daemon=True)
        self.monitor_thread.start()

        # 创建 Dask 客户端，连接到指定的调度器
        if self.scheduler_address:
            client = Client(address=self.scheduler_address)
        else:
            client = Client()

        try:
            futures = []

            for i in range(self.num_runs):

                print(f"Scheduling model:{self.model} process:{i + 1}/{self.num_runs}")
                save_name=f'{i+1}'
                if save_name in self.state.status.keys():
                    state=self.state.status[save_name]["state"]
                    if state.state=="finished":
                        print("save_name:",save_name," is Already finished")
                        continue
                    # 为每个运行创建独立的模型实例
                model = self.create_model(i+1)
                if save_name in self.state.status.keys():
                    state = self.state.status[save_name]["state"]
                    model.set_state(state)
                    model.load_state()
                    #提取模型的运行文件,无实际用途
                    state.save(model.get_state_filename())
                #print("path:",model.get_state_filename())
                task = delayed(execute_training_run)(model,"dask_trainning_run",self.state_queue_name,self.scheduler_address,self.upload_interval)
                future = client.compute(task)
                # 添加回调函数以实时更新进度
                future.add_done_callback(self.create_callback())
                futures.append(future)

            # 等待所有任务完成
            client.gather(futures)

            # 运行结束后调用总结方法
            self.summarize_results()
            self.record_progress()
        finally:

            self.stop_event.set()
            if self.monitor_thread:
                self.monitor_thread.join()
            # 确保客户端在运行结束后被关闭
            client.close()
            print(f"{self.model.name} 全部迭代运行结束")


    def create_model(self, run_id):
        '''
        创建一个独立的模型实例，用于单个训练运行
        '''
        # 确保每个模型实例都有独立的状态文件
        save_name = f"{run_id}"
        model=copy.deepcopy(self.model)
        model.save_name = save_name
        return model

    def create_callback(self):
        '''
        创建一个回调函数，用于在 Future 完成时更新进度
        '''

        def callback(future):
            try:
                result_dic = future.result()
                result=result_dic["result"]
                save_name=result_dic["save_name"]
                state=result_dic["state"]
                with self.lock:
                    if save_name not in self.state.status.keys():
                        self.state.status[save_name] = {"result": None, "state": None}
                    if self.state.status[save_name]["state"] is not None:
                        state1=self.state.status[save_name]["state"]
                        if state1>state:
                            print("callback 接受到的state文件不是最新的，因此跳过")
                            return

                    flag=self.save_state_to_local(save_name,state)
                    if flag:
                        self.state.status[save_name]["result"] = result
                        self.state.status[save_name]["state"]=state
                        self.state.current_run += 1
                        self.record_progress()

                    else:
                        print("回调函数保存状态失败！")
            except Exception as e:
                print(f"回调函数处理出错: {e}")

        return callback

    def monitor_progress(self):
        '''
        轮询读取上传的状态文件并保存到本地。
        '''
        try:
            # 使用已有的 Dask 客户端创建 Queue 实例
            state_queue = Queue(name=self.state_queue_name)
        except Exception as e:
            print(f"无法创建队列 {self.state_queue_name}: {e}")
            return

        while not self.stop_event.is_set():
            try:
                # 尝试从队列中获取一个项目，设置超时为2秒
                dic=state_queue.get(timeout=self.monitor_queue_get_timeout)
                print("Monitor get:",dic["save_name"])
                save_name=dic["save_name"]
                state=dic["state"]

                with self.lock:
                    if save_name not in self.state.status.keys():
                        self.state.status[save_name]={"result":None,"state":None}
                    if self.state.status[save_name]["state"] is not None:
                        state1=self.state.status[save_name]["state"]
                        if state1>state:
                            print("Monitor 接受到的state文件不是最新的，因此跳过")
                            continue

                    flag=self.save_state_to_local(dic["save_name"],state)
                    if flag:
                        self.state.status[save_name]["state"]=state
                        print("save_name:",save_name," state_cur:",state.parameters["current_iteration"])
                        self.record_progress()
                        print(f"Monitor update savename:", dic["save_name"]," data ok!")
                    else:
                        print(f"Monitor update savename:", dic["save_name"], " data error")
            except asyncio.exceptions.TimeoutError as a:
                # 队列为空，稍后重试
                pass
            except Exception as e:
                print(f"监控线程读取队列时出错: {e}",type(e))
            finally:
                # 每隔k秒检查一次
                time.sleep(self.monitor_sleep_interval)

    def save_state_to_local(self,save_name, state):
        '''
        将状态 pickle 保存到本地目录。
        :param save_name:
        :param state:
        :return: 是否保存成功
        '''
        filename = self.model.get_state_filename(save_name);
        save_path = filename
        save_path_tmp = save_path + ".tmp"
        try:
            # 定义保存路径，按时间戳或唯一标识符命名
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            state.save(save_path_tmp)
            os.replace(save_path_tmp, save_path)
            print(f"已保存上传的状态文件到 {save_path}")
        except Exception as e:
            if os.path.exists(save_path_tmp):
                os.remove(save_path_tmp)  # 清理临时文件
            print(f"DaskTrainingRun 保存状态文件到本地时出错: {e}")
            return False
        return True


    def summarize_results(self):
        # 汇总所有训练运行的结果，支持任意指标
        print("DaskTrainingRun summarize_results")
class TrainingPlan:
    def __init__(self, name,directory=''):
        self.name = name
        self.training_runs = []
        self.directory =directory
        self.ensure_directory()
    def ensure_directory(self):
        if self.directory and not os.path.exists(self.directory):
            os.makedirs(self.directory)
    def add_training_run(self, training_run):
        self.training_runs.append(training_run)
    def set_directory(self, directory):
        self.directory = directory
    def run(self):
        for training_run in self.training_runs:
            training_run.set_directory(self.directory)
            training_run.run()
    def get_all_progress(self):
        progress_list = []
        for run in self.training_runs:
            progress = run.get_progress()
            progress_list.append(progress)
        return progress_list
    def get_all_results(self):
        results={}
        for run in self.training_runs:
            key = f"{self.name}_{run.model.name}_{run.data}"
            results[key] = run.get_results()
        return results
class DaskTrainingPlan(TrainingPlan):
    def __init__(self, name, directory='', scheduler_address=None):
        super().__init__(name, directory)
        self.scheduler_address = scheduler_address

    def run(self):
        """
        使用 Dask 分布式调度器并行运行所有训练任务。
        """
        # 创建或连接到 Dask 客户端
        if self.scheduler_address:
            client = Client(address=self.scheduler_address)
        else:
            # 如果未指定调度器地址，则创建本地集群
            cluster = LocalCluster(n_workers=4, threads_per_worker=1, processes=False, memory_limit='4GB', scheduler_port=8789)
            client = Client(cluster)
            self.scheduler_address = cluster.scheduler_address
            print(f"Cluster 地址: {self.scheduler_address}")

        tasks = []
        for training_run in self.training_runs:
            # 设置训练任务的目录
            training_run.set_directory(self.directory)
            # 使用 Dask 的 delayed 将 run 方法封装为延迟任务
            task = delayed(training_run.run)()
            tasks.append(task)

        if tasks:
            # 提交所有任务到 Dask 客户端
            futures = client.compute(tasks)
            # 等待所有任务完成并收集结果
            client.gather(futures)

        # 关闭 Dask 客户端
        client.close()

    def get_all_progress(self):
        """
        获取所有训练计划的进度。返回一个字典，键为计划名，值为计划进度。
        """
        progress_dict = {}
        # 获取所有训练运行的进度
        for run in self.training_runs:
            progress = run.get_progress()
            progress_dict[f"{self.name}_{run.model.name}_{run.data}"] = progress
        return progress_dict

    def get_all_results(self):
        """
        获取所有训练计划的结果。返回一个字典，键为计划名_模型名_数据集，值为训练结果。
        """
        results = {}
        for run in self.training_runs:
            key = f"{self.name}_{run.model.name}_{run.data}"
            results[key] = run.get_results()
        return results

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
class TrainingManager:
    def __init__(self, directory=''):
        self.training_plans = {}
        self.directory = directory
        self.ensure_directory()
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)  # 根据需求调整线程池大小

    def ensure_directory(self):
        if self.directory and not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def add_training_plan(self, training_plan):
        """
        添加训练计划到管理器中。线程安全。
        """
        with self.lock:
            training_plan.set_directory(os.path.join(self.directory, training_plan.name))
            if not os.path.exists(training_plan.directory):
                os.makedirs(training_plan.directory)
            self.training_plans[training_plan.name] = training_plan

    def run_training_plan(self, plan_name):
        """
        运行指定名称的训练计划。线程安全。
        """
        with self.lock:
            plan = self.training_plans.get(plan_name)

        if plan:
            # 提交训练计划的运行到线程池
            future = self.executor.submit(plan.run)
            return future
        else:
            print(f"没有名为 {plan_name} 的训练计划")
            return None

    def list_training_plans(self):
        """
        列出所有训练计划的名称。线程安全。
        """
        with self.lock:
            return list(self.training_plans.keys())

    def get_all_progress(self):
        """
        获取所有训练计划的进度。线程安全。
        返回一个字典，键为计划名，值为计划进度。
        """
        progress_dict = {}
        with self.lock:
            # 获取所有训练计划的名称和实例
            plans = list(self.training_plans.items())

        # 并发获取每个训练计划的进度
        futures = {self.executor.submit(plan.get_all_progress): name for name, plan in plans}
        for future in as_completed(futures):
            name = futures[future]
            try:
                progress = future.result()
                progress_dict[name] = progress
            except Exception as e:
                print(f"获取计划 {name} 的进度时出错: {e}")
                progress_dict[name] = None

        return progress_dict

    def get_all_results(self):
        """
        获取所有训练计划的结果。线程安全。
        """
        results = {}
        with self.lock:
            plans = list(self.training_plans.values())

        # 并发获取每个训练计划的结果
        futures = {self.executor.submit(plan.get_all_results): plan for plan in plans}
        for future in as_completed(futures):
            plan_results = future.result()
            results.update(plan_results)

        return results

    def shutdown(self):
        """
        关闭线程池，等待所有正在执行的任务完成。
        """
        self.executor.shutdown(wait=True)


def test():
    # 定义模型
    model1 = Model('Model1',None)
    model2 = Model('Model2',None)
    model3 = Model('Model1',None)
    # 定义数据集
    data1 = 'Data1'
    data2 = 'Data2'

    # 创建训练计划
    plan = TrainingPlan('Plan1')

    # 指定目录
    directory = "train_test"

    # 添加训练任务，在构造函数中传入目录
    run1 = TrainingRun(model1, data1, num_runs=5, plan_name=plan.name, directory=directory)
    run2 = TrainingRun(model2, data2, num_runs=3, plan_name=plan.name, directory=directory)
    run3 = TrainingRun(model3, data2, num_runs=4, plan_name=plan.name, directory=directory)

    plan.add_training_run(run1)
    plan.add_training_run(run2)
    plan.add_training_run(run3)

    # 创建训练管理器并添加计划
    manager = TrainingManager(directory)
    manager.add_training_plan(plan)

    # 运行训练计划
    manager.run_training_plan('Plan1')
    time.sleep(10)
    # 获取训练进度和结果
    all_progress = manager.get_all_progress()
    all_results = manager.get_all_results()

    print("训练进度:")
    print(all_progress)

    print("\n训练结果:")
    for key, results in all_results.items():
        print(f"{key}: {results}")

def test_dask():
    from dask.distributed import Client, LocalCluster

    # 创建本地 Dask 集群
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='4GB', scheduler_port=8789)

    # 获取本地 cluster 的地址
    scheduler_address = cluster.scheduler_address
    print(f"Cluster 地址: {scheduler_address}")

    # 定义数据集
    data1 = 'Data1'
    data2 = 'Data2'

    # 定义模型，使用 DaskModel，并传递调度器地址
    model1 = DaskModel('Model1', None, scheduler_address=scheduler_address)
    model2 = DaskModel('Model2', None, scheduler_address=scheduler_address)

    # 创建训练计划
    plan = TrainingPlan('Plan1')

    # 指定目录
    directory = "train_test"

    # 添加训练任务，使用 DaskTrainingRun，并传递调度器地址
    run1 = DaskTrainingRun(model1, data1, num_runs=5, plan_name=plan.name, directory=directory, scheduler_address=scheduler_address)
    run2 = DaskTrainingRun(model1, data2, num_runs=3, plan_name=plan.name, directory=directory, scheduler_address=scheduler_address)
    run3 = DaskTrainingRun(model2, data1, num_runs=4, plan_name=plan.name, directory=directory, scheduler_address=scheduler_address)

    plan.add_training_run(run1)
    plan.add_training_run(run2)
    plan.add_training_run(run3)

    # 创建训练管理器并添加计划
    manager = TrainingManager(directory)
    manager.add_training_plan(plan)

    # 运行训练计划
    manager.run_training_plan('Plan1')


    while 1:
        # 获取训练进度和结果
        all_progress = manager.get_all_progress()
        all_results = manager.get_all_results()
        print("训练进度:",all_progress)

        print("\n训练结果:")
        for key, results in all_results.items():
            print(f"{key}: {results}")
        time.sleep(10)
    # 关闭 Dask 集群
    cluster.close()

def test_dask_training_run():
    from dask.distributed import Client, LocalCluster

    # 创建本地 Dask 集群
    cluster = LocalCluster(n_workers=4, threads_per_worker=1, memory_limit='4GB', scheduler_port=8787)

    # 获取本地 cluster 的地址
    scheduler_address = cluster.scheduler_address
    print(f"Cluster 地址: {scheduler_address}")

    model1 = Model('Model1', None)
    # 定义数据集
    data1 = ModelData('data1', None)

    # 指定目录
    directory = "train_test1"

    client = Client(scheduler_address)
    # 添加训练任务，使用 DaskTrainingRun，并传递调度器地址
    run1 = DaskTrainingRun(model1, data1, num_runs=5, plan_name="", directory=directory,
                           scheduler_address=scheduler_address)
    run1.run()
    while 1:
        time.sleep(10)
# 示例用法
if __name__ == "__main__":
   test_dask_training_run()
