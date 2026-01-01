# [关键] 数据记录容器 (Recorder)

from abc import ABC, abstractmethod
from typing import List, Any
import copy

class IDebugger(ABC):
    """调试器接口"""
    @abstractmethod
    def record_open_set_node(self, node: Any): pass
    
    @abstractmethod
    def record_current_expansion(self, node: Any): pass
    
    @abstractmethod
    def set_cost_map(self, cost_map: Any): pass

class NoOpDebugger(IDebugger):
    """[工业界技巧] 空对象模式
    用于生产环境或大规模测试。它的所有方法都是空的，
    Python 的函数调用开销极小，这样算法代码里就不用写 if debug: 了。
    """
    def record_open_set_node(self, node: Any): pass
    def record_current_expansion(self, node: Any): pass
    def set_cost_map(self, cost_map: Any): pass

class PlanningDebugger(IDebugger):
    """真正的记录器
    用于开发和演示。会真的消耗内存去存数据。
    """
    def __init__(self):
        self.open_set_history = []
        self.expanded_nodes = []
        self.cost_map = None

    def record_open_set_node(self, node: Any):
        # 注意：这里可能需要深拷贝，防止后续修改影响历史记录
        # 但在 Python 中为了速度，通常只存 (x, y) 坐标元组
        self.open_set_history.append((node.x, node.y))

    def record_current_expansion(self, node: Any):
        self.expanded_nodes.append((node.x, node.y))

    def set_cost_map(self, cost_map: Any):
        self.cost_map = cost_map # 引用即可，因为地图一般不变