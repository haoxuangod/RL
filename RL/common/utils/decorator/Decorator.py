

class ParameterError(Exception):
    """Exception raised for errors in the parameter input."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
def Builder(cls):
    '''
    1.该装饰器会自动创建构造函数__init__完成初始化数据，
    参数来源为该类中名为params的列表，格式为[(参数名，参数类型,参数默认值,是否是必须的)，]，
    假设参数名为test，参数类型为Myclass那么在__init__函数中复制属性test=Myclass(test),
    在__init__函数后调用_init函数进行进一步的初始化
    2.创建一个builder类实现建造者模式，增加setxxx(xxx为参数)
    :param cls:
    :return:
    '''
    params = getattr(cls, 'params', [])

    def __init__(self, **kwargs):
        for param_name, param_type, default_value, is_necessary in params:
            if param_name in kwargs:
                setattr(self, param_name, param_type(kwargs[param_name]))
            elif default_value is not None:
                setattr(self, param_name, param_type(default_value))
            elif is_necessary:
                raise ParameterError(f"Missing necessary parameter: {param_name}")
        self._init()

    cls.__init__ = __init__

    original_init = cls.__dict__.get('_init', lambda self: None)

    def _init(self):
        original_init(self)

    cls._init = _init

    # 创建 Builder 类
    class BuilderClass:
        def __init__(self):
            self.values = {}

        def set(self, name, value):
            self.values[name] = value
            return self

        def build(self):
            # 检查所有必要参数是否存在
            for param_name, param_type, default_value, is_necessary in params:
                if is_necessary and param_name not in self.values and default_value is None:
                    raise ParameterError(f"Missing necessary parameter: {param_name}")
            return cls(**self.values)

    # 动态创建 setXxx 方法
    for param_name1, param_type, default_value, is_necessary in params:
        setattr(BuilderClass, f'set{param_name1.capitalize()}',
                lambda self, value, param_name=param_name1: self.set(param_name, value))

    cls.builder = BuilderClass
    return cls

# 示例使用
class MyClass:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"MyClass(value={self.value})"


@Builder
class Example:
    params = [('test', MyClass, None,True), ('number', int, 0,True),('abc',float,None,True)]

    def _init(self):
        print("Further initialization")

def main():
    # 测试
    try:
        builder = Example.builder()
        example_instance = builder.setTest(10).setNumber(42).build()
        print(example_instance.test)  # MyClass(value=10)
        print(example_instance.number)  # 42
    except ParameterError as e:
        print(e.message)

    try:
        builder = Example.builder()
        example_instance = builder.setNumber(42).build()
        print(example_instance.test)
        print(example_instance.number)
    except ParameterError as e:
        print(e.message)  # Missing necessary parameter: test

if __name__=='__main__':
    main()
