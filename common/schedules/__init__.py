from abc import ABC, abstractmethod

class Schedule(ABC):
    """
    抽象调度器基类，定义 alpha 等参数如何随时间衰减。
    """

    @abstractmethod
    def value(self, step: int) -> float:
        """
        返回当前 step 对应的值（例如 alpha、epsilon 等）。
        """
        pass
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
