import importlib
import inspect
import json
import os
import sys
import warnings
import dill
import re
from abc import ABC, abstractmethod
from collections import deque
import queue

import pkgutil
def load_all_modules(root_path):
    """递归加载项目目录下所有模块"""

    if not os.path.exists(root_path):
        print(f"模块不存在:{root_path}")
        return
    if os.path.isfile(root_path):
        try:
            paths=root_path.split(".")
            path='.'.join(paths[:-1])
            importlib.import_module(path)
        except ImportError as e:
            print(f"加载模块失败: {root_path}, 错误: {e}")
    else:
        for loader, module_name, is_pkg in pkgutil.walk_packages([root_path]):
            full_module_name = f"{root_path}.{module_name}"
            if is_pkg:
                # 递归处理子包
                load_all_modules(full_module_name)
            else:
                try:
                    importlib.import_module(full_module_name)
                except ImportError as e:
                    print(f"加载模块失败: {full_module_name}, 错误: {e}")


class ClassNotFoundException(Exception):
    pass
class SubclassTrackerMeta(type):
    '''
    以该类为元类的对象均会被记录到_registry中
    '''
    _registry = {}
    _alias={}
    def __new__(mcs, name, bases, attrs):
        # 创建类对象
        cls = super().__new__(mcs, name, bases, attrs)
        if not "_registry" in mcs.__dict__:
            mcs._registry = {}  # 每个元类实例独立存储
        if not "_alias" in mcs.__dict__:
            mcs._alias = {}

        # 仅处理非元类的普通类
        if not issubclass(cls, type):
            # 确保每个元类实例有独立注册表
            if not hasattr(mcs, '_registry'):
                mcs._registry = {}
            # 记录类名与类的映射
            mcs._registry[name] = cls
            mcs._alias[name]=name
            #将类的别名与真实名称一一对应起来
            if "alias" in cls.__dict__ and isinstance(cls.__dict__["alias"],(list,tuple,set)):
                for alias in cls.__dict__["alias"]:
                    if alias in mcs._alias:
                        warnings.warn(f"The alias of class {mcs._alias[alias]} in meta {mcs} already exists so it will not be valid")
                    else:
                        mcs._alias[alias]=name


        return cls

    @classmethod
    def get_registry(cls):
        return cls._registry
    @classmethod
    def get_class(cls,name):
        try:
            return cls._registry[cls._alias[name]]
        except Exception:
            raise ClassNotFoundException(f"Class:{name} not found in class registry")

class RegexMeta(SubclassTrackerMeta):
    '''
    类必须有pattern属性，用于构造函数中输入字符串参数后正则表达式匹配判断是否满足类条件
    该类会记录所有子类，可用于工厂模式
    '''

    def __new__(cls, name, bases, class_dict):
        pattern = class_dict.get('pattern')
        if pattern is not None:
            if isinstance(pattern,str):
                pattern=[pattern,]
            new_class = super().__new__(cls, name, bases, class_dict)
            new_class.pattern=[]
            for p in pattern:
                new_class.pattern.append(re.compile(p))
        else:
            raise ValueError(f"Class {name} must have a 'pattern' attribute.")
        return new_class


class ParameterError(Exception):
    """Exception raised for errors in the parameter input."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
class BuilderMeta(type):

    '''
    1.该元类会自动创建构造函数__init__完成初始化数据，
    参数来源为该类中名为params的列表，格式为[(参数名，参数类型,参数默认值,是否是必须的)，]，
    假设参数名为test，参数类型为Myclass那么在__init__函数中复制属性test=Myclass(test),
    在__init__函数后调用_init函数进行进一步的初始化
    2.创建一个builder类实现建造者模式，增加setxxx(xxx为参数)


    '''
    def __new__(cls, name, bases, attrs):
        # 如果当前类定义了自己的 params，则使用它；否则从父类继承
        if len(bases)!=0:
            params = attrs.get('params', getattr(bases[0], 'params', []))
        else:
            params = attrs.get('params',[])
        #print(name,params)
        # 添加 __init__ 方法
        def __init__(self, **kwargs):
            for param_name, param_type, default_value, is_necessary in params:
                if param_name in kwargs:
                    setattr(self, param_name, kwargs[param_name])
                elif default_value is not None:
                    setattr(self, param_name, default_value)
                elif is_necessary:
                    raise ParameterError(f"Missing necessary parameter: {param_name}")
            self._init()

        attrs['__init__'] = __init__

        # 添加 _init 方法
        original_init = attrs.get('_init', lambda self: None)
        def _init(self):
            original_init(self)
        attrs['_init'] = _init

        # 创建 Builder 类
        class BuilderClass:
            def __init__(self):
                self.values = {}

            def set(self, name, value):
                self.values[name] = value
                return self

            def build(self):
                for param_name, param_type, default_value, is_necessary in params:
                    if is_necessary and param_name not in self.values and default_value is None:
                        raise ParameterError(f"Missing necessary parameter: {param_name}")
                return cls(**self.values)

        # 动态创建 setXxx 方法
        for param_name1, param_type, default_value, is_necessary in params:
            setattr(BuilderClass, f'set{param_name1.capitalize()}',
                    lambda self, value, param_name=param_name1: self.set(param_name, value))

        attrs['builder'] = BuilderClass

        return super().__new__(cls, name, bases, attrs)


def get_main_module_name():
    main_module = sys.modules['__main__']
    package = getattr(main_module, '__package__', None)
    if hasattr(main_module, '__file__'):
        file_path = main_module.__file__
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if base_name == '__init__':
            if package:
                module_name = package
            else:
                dir_path = os.path.dirname(file_path)
                dir_name = os.path.basename(dir_path)
                module_name = dir_name
        else:
            if package:
                module_name = f"{package}.{base_name}"
            else:
                module_name = base_name
    return module_name


def get_object_module_name(obj):
    module_name = obj.__module__
    if module_name == '__main__':
        module_name = get_main_module_name()

    return module_name

'''
def get_class_from_type_str(type_str):
    #还没有修改完成
    # 提取类路径
    match = re.search(r"'(.+?)'", type_str)
    if not match:
        raise ValueError("Invalid type string format")
    full_name = match.group(1)

    # 分割模块路径和类名
    if '.' in full_name:
        module_path, class_name = full_name.rsplit('.', 1)
    else:
        module_path = 'builtins'  # 处理内置类型
        class_name = full_name

    # 导入模块
    if module_path == '__main__':
        module_name = get_main_module_name()
        module = sys.modules[module_name]
    else:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            raise ImportError(f"Module '{module_path}' not found")

    # 获取类
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")
    return cls
'''
class ValidationError(Exception):
    '''

    '''
    pass
class SerializeError(Exception):
    pass
class Serializer:
    '''
    序列化任意已知模块对象的基类
    '''
    #支持的类的列表
    supported_types = []
    def serialize(self, obj):
        if isinstance(obj,tuple(self.supported_types)):
            return self._serialize(obj)
        else:
            raise SerializeError(f"序列化不支持的类型：{obj}")

    def _serialize(self, obj):
        raise NotImplementedError
    def deserialize(self,cls,data):
        if issubclass(cls, tuple(self.supported_types)):
            return self._deserialize(cls,data)
        else:
            raise SerializeError(f"反序列化不支持的类型：{cls}")
    def _deserialize(self,obj,data):
        raise NotImplementedError
    


def has_parameter(func, param_name):
    # 获取函数/方法的参数签名
    try:
        sig = inspect.signature(func)
    except ValueError:
        # 处理内置函数等无法获取签名的特殊情况
        return False

    # 检查参数列表中是否存在目标参数
    return param_name in sig.parameters

def has_parameters(func,param_names):

    for param in param_names:
        if not has_parameter(func,param):
            return False
    return True
class SerializeTemp:
    '''
    用于判断是否是已经标记过正在/已经序列化完成的对象,目的是防止循环嵌套对象导致无限递归
    '''
    def __init__(self,cnt):
        self.cnt=cnt
    def __eq__(self, other):
        return self.cnt == other.cnt
    def __hash__(self):
        return self.cnt

'''
2025.04.09
1.需要解决使用global导致的多线程冲突问题 √
2.from_dict中对子类的from_dict的调用 √
3.将不同的对象中替换为SerialTemp对象的优秀实现。
4.serializer的覆盖优先级（对相同类的序列化 子类定义的应该覆盖父类的）

2025.04.21
1.需要解决字典对象的键序列化问题（用SerialTemp对象给代替，一个列表记录然后回头替换掉）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
'''


def from_dict(data):
    if not "__type__" in data.keys():
        raise ValidationError("字典没有表明类型的__class__键")
    if "__dict__" not in data.keys():
        raise ValidationError("字典没有__dict__键")

    try:
        tmp = data["__type__"].split(".")
        class_name = tmp[-1]
        module_name = ".".join([data for data in tmp[:-1]])
        module = importlib.import_module(module_name)
        cls_load = getattr(module, class_name)
    except Exception:
        cls_context = data["__type__"]
        raise ValidationError(f"无法导入__type__={cls_context} 的类")
    if hasattr(cls_load, "from_dict"):

        if "additional_lst" in data:
            return cls_load.from_dict(data["__dict__"],additional_lst=data["additional_lst"])
        return cls_load.from_dict(data["__dict__"])
    else:
        raise ValidationError("类没有from_dict方法")

class SerializeMeta(type):

    def __new__(cls, name, bases, attrs):
        '''
        对类增加若干属性：
        1.serializers_support_types:在序列化当前类中父类和自己的serializers支持的所有对象类型
        2.support_types:在序列化当前类中支持的所有对象类型
        3.serializers:自己和父类定义的serializers的总和（已去重列表)
        4.serializers_dict:{serializer.__class__.__name__:serializer for serializer in attrs['serializers'] }
        5.serializers_cls_dic:只有在父类或自己有定义Serializers时才会有该属性，{cls:[Serializers]}以类为键，得到按优先级排序的serializers列表，
        用于在一个类有多个serializers可以序列化时选择最优先的

        增加的方法：
        1.to_dict():将对象序列化为字典
            dic["__type__"] =  f"{get_object_module_name(obj)}.{obj.__class__.__name__}"
            dic["__dict__"] =  {attribute_name：serialize(value)}
            dic['cnt']=0(根节点序列化顺序为0)

            在__dict__中对于属性的不同的对象类型(相同类型对象按优先级排序)：
            键cnt表示属性值的序列化顺序
            [1]内置对象 int,float,str,bool,None直接保留：
                dic["__dict__"] = {attribute1:5,attribute2：“hello_world”}
            [2]对于定义在了serializers中的对象：
                dic["__dict__"]={attribute1:dic1}
                dic1={"__type__": f"{get_object_module_name(obj)}.{obj.__class__.__name__}",
                                    'data':serializer.serialize(obj),
                                    'serializer':serializer.__class__.__name__
                                    'cnt':cnt}
            [3]含有to_dict方法的对象（无论是不是当前元类的子类）直接调用to_dict方法然后序列化
                dic["__dict__"] =  {attribute1：to_dict(值1)}
            [4]集合对象:list,dict,tuple将每个元素序列化后放入:
                dic["__dict__"]={attribute_list:[serialize(obj) for obj in list]}
                dic["__dict__"]={attribute_tuple:(serialize(item) for item in obj)}
                字典：
                如果k是基础类型：
                dic1={k: serialize(v) for k, v in obj.items()}
                否则如果k是复杂对象则将k给序列化后加入到additional_dic中，并且使用SerialTemp对象来代替:
                dic1={SerializeTemp(cnt):serialize(v) for k, v in obj.items()}
                dic["__dict__"]={attribute_dic:dic1}

            [5]自带数据结构集合对象:deque、set、queue.Queue、queue.PriorityQueue:
                dic["__dict__"]={deque_attribute:deque_dic,set_attribute:set_attribute_dic
                                 queue_attribute:queue_dic,priority_queue_attribute:priority_queue_dic}
                deque_dic={ '__type__': 'deque',
                        'data': [serialize(item) for item in obj],
                        'maxlen': obj.maxlen,
                        'cnt':cnt}
                set_dic={
                            '__type__': 'set',
                            'data': [serialize(item) for item in obj],
                            ‘cnt’:cnt
                        }
                queue_dic={
                    '__type__': 'queue',
                    'data': [serialize(item) for item in obj.queue],
                    'maxsize':obj.maxsize,
                    'cnt': cnt1
                }
                priority_queue_dic={
                    '__type__': 'priority_queue',
                    'data': [serialize(item) for item in obj.queue],
                    'cnt':cnt1
                }


            [6]不含有to_dict方法、无法被serializers序列化的对象
                dic["__dict__"]={attribute1:{__dill__:str(dill.dumps(obj).hex()),
                                             "cnt":cnt}}


        2.类方法 from_dict(dic)：从字典中加载对象
            维护两个字典：
            cnt_dic = {cnt:obj} 用于记录不同cnt对应的已经序列化的对象
            obj_dic = {id(obj):[(obj,key,cnt,func),]} 用一个列表记录obj的是SerialTemp的属性（容器key就是索引），
            代入cnt_dic[cnt]得到实际对象（整个序列化过程完成后），然后通过func来修改obj如：
            
            设置对象属性
            def attribute_func(obj1, key, value):
                setattr(obj1, key, value)
            set修改元素
            def set_func(obj1, key, value):
                obj1.remove(key)
                obj1.add(value)
            queue.Queue修改元素，单次访问O（N）太慢 要修改元素数量多了不如重新构建一遍队列（可优化地方）
            def queue_func(obj1, key, value):
                obj1.queue[key]=value
            queue.PriorityQueue的内部存储是list所以直接修改O(1)没问题
            def priority_queue_func(obj1, key, value):
                obj1.queue[key]=value


        '''

        # 辅助函数：检查类或其祖先是否显式定义了某个属性
        def is_defined_in_hierarchy(klass, attr_name):
            # 遍历类的 MRO（包括所有基类），检查是否直接定义了属性
            for base in klass.__mro__:
                if attr_name in base.__dict__:
                    return True
            return False

        to_dict_defined = any(is_defined_in_hierarchy(base, 'to_dict') for base in bases)
        from_dict_defined = any(is_defined_in_hierarchy(base, 'from_dict') for base in bases)
        serializers_defined = any(is_defined_in_hierarchy(base, 'serializers') for base in bases)
        serializers_defined=serializers_defined or 'serializers' in attrs
        support_types = {queue.PriorityQueue,queue.Queue,deque, list, dict, tuple, set, int, float, str, bool, type(None)}
        serializers_support_types = set()
        # 动态添加方法（仅当当前类和祖先均未定义时）
        if not to_dict_defined and 'to_dict' not in attrs:
            attrs['to_dict'] = cls._create_to_dict_method()
        if not from_dict_defined and 'from_dict' not in attrs:
            attrs['from_dict'] = cls._create_from_dict_method()
        if not serializers_defined and 'serializers' not in attrs:
            attrs['serializers'] = []
        elif serializers_defined:
            if 'serializers' not in attrs:
                attrs['serializers'] =[]
            serializers=set(attrs['serializers'])
            '''
            无法直接获取到当前类因为还没有创建，cls是当前元类
            '''
            for base in bases:
                for base1 in base.__mro__:
                    if "serializers" in base1.__dict__:
                        serializers.update(base1.__dict__["serializers"])
            for serializer in serializers:
                serializers_support_types.update(serializer.supported_types)
            support_types.update(serializers_support_types)
            # 存储不同类型对应的serializer优先级排序
            serializers_cls_dic = {support_type:[] for support_type in serializers_support_types}
            for serializer in attrs["serializers"]:
                for support_type in serializer.supported_types:
                    serializers_cls_dic[support_type].append(serializer)
            attrs['serializers'] = list(serializers)
            for base in bases:
                for base1 in base.__mro__:
                    if "serializers" in base1.__dict__:
                        for serializer in base1.__dict__["serializers"]:
                            for support_type in serializer.supported_types:
                                serializers_cls_dic[support_type].append(serializer)
            attrs['serializers_cls_dic'] = serializers_cls_dic
        exclude_set=set()

        if "__exclude__" in attrs:
            exclude_set.update(attrs["__exclude__"])
        for base in bases:
            for base1 in base.__mro__:
                if hasattr(base1,"__exclude__"):
                    exclude_set.update(base1.__exclude__)

        exclude_class_set=set()
        exclude_attr_set=set()
        for data in exclude_set:
            if inspect.isclass(data):
                exclude_class_set.add(data)
            else:
                exclude_attr_set.add(data)

        attrs['exclude_class_set'] = exclude_class_set
        attrs['exclude_attr_set'] = exclude_attr_set

        attrs['support_types'] = support_types
        attrs['serializers_support_types'] = serializers_support_types
        attrs['serializers_dict']={serializer.__class__.__name__:serializer for serializer in attrs['serializers'] }
        return super().__new__(cls, name, bases, attrs)

    @classmethod
    def _create_to_dict_method(cls):

        def to_dict(self,vis_dict={},cnt=[],additional_lst=[],is_root=True):
            '''

            :param self:
            :param vis_dict:
            :param cnt:
            :param additional_lst: 只有root才有用的属性由from_dict方法给出
            :param is_root:
            :return:
            '''

            global id
            if is_root:
                cnt=[0]
                vis_dict={id(self):{'__dill__':str(dill.dumps(SerializeTemp(cnt[0])).hex())}}

            def serialize(obj):
                if isinstance(obj,tuple(self.__class__.exclude_class_set)):
                    return None
                nonlocal cnt
                if not isinstance(obj,(type(None),int,float,str,bool)):
                    if id(obj) not in vis_dict.keys():
                        cnt[0]=cnt[0]+1
                        vis_dict[id(obj)]={'__dill__':str(dill.dumps(SerializeTemp(cnt[0])).hex())}
                    else:
                        return vis_dict[id(obj)]
                cnt1=cnt[0]
                support_types=self.support_types

                if not isinstance(obj,tuple(support_types)) and not hasattr(obj, 'to_dict'):
                    try:
                        return {'__dill__': str(dill.dumps(obj).hex()),
                                'cnt':cnt1}
                    except Exception as e:

                        raise SerializeError(f"Can't serialize obj:{obj} , obj is not in supported_types"
                              f"types: {support_types} and can not dill it \nReason:{e.__class__.__name__}:{e}")

                elif isinstance(obj,tuple(self.serializers_support_types)):
                    key=None
                    for tp in self.serializers_support_types:
                        if isinstance(obj,tp):
                            key=tp
                            break
                    serializers_cls_dic = self.serializers_cls_dic
                    #最新的类定义的serializer序列化obj
                    serializer=serializers_cls_dic[key][0]
                    #应该保留类信息以便于反序列化
                    return {"__type__": f"{get_object_module_name(obj)}.{obj.__class__.__name__}",
                                    'data':serializer.serialize(obj),
                                    'serializer':serializer.__class__.__name__,
                                    "cnt":cnt1}


                if isinstance(obj, deque):
                    return {
                        '__type__': 'deque',
                        'data': [serialize(item) for item in obj],
                        'maxlen': obj.maxlen,
                        'cnt':cnt1
                    }
                elif isinstance(obj, set):
                    return {
                        '__type__': 'set',
                        'data': [serialize(item) for item in obj],
                        'cnt':cnt1
                    }
                elif isinstance(obj,queue.PriorityQueue):
                    return {
                        '__type__': 'priority_queue',
                        'data': [serialize(item) for item in obj.queue],
                        'cnt': cnt1
                    }

                elif isinstance(obj,queue.Queue):
                    return {
                        '__type__': 'queue',
                        'data': [serialize(item) for item in obj.queue],
                        'maxsize': obj.maxsize,
                        'cnt': cnt1
                    }

                # 自定义对象优先调用 to_dict
                elif hasattr(obj, 'to_dict'):
                    if has_parameters(obj.to_dict,('is_root',"cnt","vis_dict","additional_lst")) :
                        dic=obj.to_dict(is_root=False,cnt=cnt,vis_dict=vis_dict,additional_lst=additional_lst)
                    else:
                        dic=serialize(obj.to_dict())
                    if "__type__" not in dic:
                        dic["__type__"] =  f"{get_object_module_name(obj)}.{obj.__class__.__name__}"
                    return dic
                # 处理列表
                elif isinstance(obj, list):
                    return [serialize(item) for item in obj]
                # 处理元组
                elif isinstance(obj, tuple):
                    return tuple(serialize(item) for item in obj)
                # 处理字典
                elif isinstance(obj, dict):
                    result={}
                    for k,v in obj.items():
                        if isinstance(k,(int, float, str, bool, type(None))):
                            result[k]=serialize(v)
                        else:

                            if id(k) in vis_dict.keys():
                                key=json.dumps(vis_dict[id(k)])
                                result[key]=serialize(v)
                            else:
                                cnt1 = cnt[0] + 1
                                k1=serialize(k)
                                key=json.dumps({'__dill__': str(dill.dumps(SerializeTemp(cnt1)).hex())})
                                result[key]=serialize(v)
                                additional_lst.append((cnt1,k1))
                    return result
                # 基本类型直接保留
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                # 其他复杂类型用 dill 序列化
                else:
                    return {'__dill__':str(dill.dumps(obj).hex()),
                            'cnt':cnt1}

            exclude_attr = getattr(self.__class__, 'exclude_attr_set', set())
            exclude_class = getattr(self.__class__, 'exclude_attr_class', set())
            data = {"__type__": f"{get_object_module_name(self)}.{self.__class__.__name__}",
                    "__dict__": {},
                    "cnt":cnt[0],
                    }
            if is_root:
                data["additional_lst"]=additional_lst
            for key, value in self.__dict__.items():
                if key in exclude_attr:
                    continue
                if value.__class__ in exclude_class:
                    continue
                data["__dict__"][key] = serialize(value)

            return data

        return to_dict

    @classmethod
    def _create_from_dict_method(cls):
        @classmethod
        def from_dict(cls, data:dict,cnt_dic={},obj_dic={},is_root=True,additional_lst=[]):

            global id
            if is_root:
                obj_dic = {}
                cnt_dic = {}

            def mapping_func(obj1, key, value):
                obj1[key] = value
            def tuple_set_func(obj1, key, value):
                obj1 = obj1[:key] + (value,) + obj1[key + 1:]
            def attribute_func(obj1, key, value):
                setattr(obj1, key, value)
            def set_func(obj1, key, value):
                obj1.remove(key)
                obj1.add(value)
            def queue_func(obj1, key, value):
                obj1.queue[key]=value
            def priority_queue_func(obj1, key, value):
                obj1.queue[key]=value
            def dic_key_func(obj1, key, value):
                val=obj1[key]
                obj1.pop(key)
                obj1[value]=val
            # 递归反序列化核心逻辑
            def deserialize(obj):

                if isinstance(obj, dict):

                    if '__type__' in obj:
                        if obj['__type__'] == 'deque':
                            datas = [deserialize(item) for item in obj['data']]
                            obj1= deque(datas, maxlen=obj['maxlen'])
                            num=0
                            for data in datas:
                                if isinstance(data,SerializeTemp):
                                    if id(obj1) not in obj_dic:
                                        obj_dic[id(obj1)]=[]
                                    obj_dic[id(obj1)].append((obj1,num,data.cnt,mapping_func))
                                num=num+1
                            return obj1
                        elif obj['__type__'] == 'set':
                            datas={deserialize(item) for item in obj['data']}
                            for data in datas:
                                if isinstance(data,SerializeTemp):
                                    if id(datas) not in obj_dic:
                                        obj_dic[id(datas)]=[]
                                    obj_dic[id(datas)].append((datas,data,data.cnt,set_func))
                            return datas
                        elif obj['__type__'] == 'queue':
                            datas = [deserialize(item) for item in obj['data']]
                            obj1 = queue.Queue(maxsize=obj['maxsize'])
                            obj1.queue=deque(datas)
                            num = 0
                            for data in datas:
                                if isinstance(data, SerializeTemp):
                                    if id(obj1) not in obj_dic:
                                        obj_dic[id(obj1)] = []
                                    obj_dic[id(obj1)].append((obj1, num, data.cnt, queue_func))
                                num = num + 1
                            return obj1

                        elif obj['__type__'] == 'priority_queue':
                            datas = [deserialize(item) for item in obj['data']]
                            obj1 = queue.PriorityQueue()
                            for data in datas:
                                obj1.put(data)
                            for i,data in enumerate(obj1.queue):
                                if isinstance(data, SerializeTemp):
                                    if id(obj1) not in obj_dic:
                                        obj_dic[id(obj1)] = []
                                    obj_dic[id(obj1)].append((obj1, i, data.cnt, priority_queue_func))

                        else:
                            class_info = obj['__type__']
                            module_name, class_name = class_info.rsplit('.', 1)
                            try:
                                module = importlib.import_module(module_name)
                                target_cls = getattr(module, class_name)
                            except (ImportError, AttributeError):
                                return obj  # 回退原始数据


                            if 'serializer' in obj:
                                serializer=cls.serializers_dict[obj['serializer']]
                                obj = serializer.deserialize(target_cls, obj["data"])
                                return obj
                            # 处理自定义对象
                            elif '__dict__' in obj:

                                try:
                                    #instance = target_cls.__new__(target_cls)

                                    if has_parameters(target_cls.from_dict, ('is_root','cnt_dic','obj_dic')):
                                        instance=target_cls.from_dict(obj["__dict__"],cnt_dic,obj_dic,False)
                                    else:
                                        instance=target_cls.from_dict(obj["__dict__"])

                                    if "cnt" in obj:
                                        cnt_dic[obj["cnt"]] =instance

                                    return instance

                                except (ImportError, AttributeError):
                                    return obj  # 回退原始数据

                    # 处理 dill 序列化的对象
                    elif '__dill__' in obj:
                        obj1=dill.loads(bytes.fromhex(obj['__dill__']))
                        if 'cnt' in obj:
                            cnt_dic[obj["cnt"]]=obj1
                        return obj1
                    # 普通字典递归处理
                    else:

                        datas={k: deserialize(v) for k, v in obj.items()}

                        if id(datas) not in obj_dic:
                            obj_dic[id(datas)] = []
                        for key,value in datas.items():
                            if isinstance(key,str) :
                                try:
                                    dic=json.loads(key)
                                    if isinstance(dic,dict) and "__dill__" in dic:
                                        obj2=dill.loads(bytes.fromhex(dic["__dill__"]))
                                        if isinstance(obj2,SerializeTemp):
                                            obj_dic[id(datas)].append((datas,key,obj2.cnt,dic_key_func))

                                except Exception as e:
                                    pass

                            if isinstance(value, SerializeTemp):
                                obj_dic[id(datas)].append((datas, key, value.cnt,mapping_func))
                        return datas
                # 处理列表
                elif isinstance(obj, list):
                    datas=[deserialize(item) for item in obj]
                    if id(datas) not in obj_dic:
                        obj_dic[id(datas)] = []
                    num=0

                    for data in datas:
                        if isinstance(data, SerializeTemp):
                            obj_dic[id(datas)].append((datas, num, data.cnt,mapping_func))
                        num=num+1
                    return datas
                # 处理元组
                elif isinstance(obj, tuple):
                    datas=tuple(deserialize(item) for item in obj)
                    num=0
                    for data in datas:
                        if isinstance(data, SerializeTemp):
                            if id(datas) not in obj_dic:
                                obj_dic[id(datas)] = []

                            obj_dic[id(datas)].append((datas, num, data.cnt,tuple_set_func))
                        num=num+1
                    return datas
                # 基本类型直接保留
                else:
                    return obj

            obj = cls.__new__(cls)  # 不调用 __init__

            if "__type__" in data.keys():
                #兼容有__type__的字典
                try:
                    tmp = data["__type__"].split(".")
                    class_name = tmp[-1]
                    module_name = ".".join([data for data in tmp[:-1]])
                    module = importlib.import_module(module_name)
                    cls_load = getattr(module, class_name)
                    cls_module_name = get_object_module_name(obj)
                except Exception:
                    cls_context = data["__type__"]
                    raise ValidationError(f"无法导入__type__={cls_context} 的类")
                if "__dict__" not in data.keys():
                    return obj
                if cls_module_name != module_name:
                    raise ValidationError(f"无法加载class:{cls} dict中的cls是{cls_load}")
                dic=data["__dict__"]
                if "additional_lst" in data:
                    lst=data["additional_lst"]
                    for (k,v) in lst:
                        cnt_dic[k]=deserialize(v)
            else:
                dic=data

            if is_root:
                cnt_dic[0] = obj
                for (k,v) in additional_lst:
                    cnt_dic[k]=deserialize(v)
            for key, value in dic.items():
                val=deserialize(value)
                if isinstance(val,SerializeTemp):
                    if id(obj) not in obj_dic.keys():
                        obj_dic[id(obj)]=[]
                    obj_dic[id(obj)].append((obj,key,val.cnt,attribute_func))
                setattr(obj, key, val)

            if is_root:
                for id1, lst in obj_dic.items():
                    for obj1,key,cnt,func in lst:
                        func(obj1,key,cnt_dic[cnt])

            # 重新初始化排除的字段（如线程）
            if hasattr(obj, '__deserialize_post__'):
                obj.__deserialize_post__()
            return obj
        return from_dict


import threading
class Banana(metaclass=SerializeMeta):
    def __init__(self,name,size=1,isgood=True):
        self.name=name
        self.size=size
        self.isgood=isgood
    def __lt__(self, other):
        return self.size<other.size
class A(metaclass=SerializeMeta):
    __exclude__ = ['thread']  # 排除线程字段

    def __init__(self, len, target,banana,c):
        self.len = len
        self.dic={c:1}
        self.buffer = deque(maxlen=len)  # 构造函数初始化空 buffer
        self.thread = threading.Thread(target=target)
        self.banana = banana
        self.target=target
        self.lst=[Banana(1),Banana(2),Banana(3)]
        self.buffer.append(Banana(4))
        self.thread.start()
        self.queue=queue.Queue(maxsize=100)
        self.queue.put(Banana(1))
        self.priority_queue=queue.PriorityQueue()
        self.priority_queue.put(Banana(1))
        self.priority_queue.put(Banana(2))
        self.priority_queue.put(Banana(3))
    def __deserialize_post__(self):
        """反序列化后重新启动线程"""
        if not hasattr(self, 'thread'):
            self.thread = threading.Thread(target=self.target)  # 假设 target 已保存
            self.thread.start()
class C:
    def __init__(self,size):
        self.size=size
    def __hash__(self):
        return self.size
    def __eq__(self, other):
        return self.size==other.size

def Serialize():

    def example_target():
        print("Thread running")
    # 创建对象并填充 buffer
    a = A(len=3, target=example_target,banana=Banana("BananaOne"),c=C(size=114514))
    a.buffer.extend([1, 2, 3])

    # 序列化为字典
    data = a.to_dict()
    print(data)  # 输出: {'len': 3, 'buffer': {'__class__': 'collections.deque', '__pickle__': '...'}}
    # 反序列化重建对象
    #new_a = A.from_dict(data)
    new_a=from_dict(data)
    print(new_a.buffer)  # 输出: deque([1, 2, 3], maxlen=3)
    print(new_a.thread)  # 输出: None（被排除，但 __deserialize_post__ 会重建）

    with open("test.json", "w") as f:
        json.dump(data,f)
if __name__ == '__main__':
    Serialize()