from Decorator.Meta import *

class A(metaclass=SubclassTrackerMeta):
    alias=["a","aaa","AAA"]

class B(A):
    alias=["a"]
    pass
class C(metaclass=SubclassTrackerMeta):
    pass
class Meta(SubclassTrackerMeta):
    pass
class D(metaclass=Meta):
    pass
class E(D):
    pass

print(A.get_registry())
print(E.get_registry())
print(C.get_class("a"))
print(A._alias)