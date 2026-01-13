# src/visualization/debugger.py
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple

class IDebugger(ABC):
    """调试器接口"""
    @abstractmethod
    def record_open_set_node(self, node: Any, f: float = 0.0, h: float = 0.0): 
        """记录加入 OpenSet 的节点及其代价"""
        pass
    
    @abstractmethod
    def record_current_expansion(self, node: Any): 
        """记录当前正在扩展的节点"""
        pass
    
    @abstractmethod
    def record_edge(self, start_node: Any, end_node: Any):
        """记录 RRT 的一条边"""
        pass
    
    @abstractmethod
    def set_cost_map(self, cost_map: Any): 
        """设置底图"""
        pass

class NoOpDebugger(IDebugger):
    """
    [工业界技巧] 空对象模式 (Null Object Pattern)
    用于生产环境。所有操作不做任何事情，且被解释器优化，开销极小。
    """
    def record_open_set_node(self, node: Any, f: float = 0.0, h: float = 0.0): pass
    def record_current_expansion(self, node: Any): pass
    def record_edge(self, start_node: Any, end_node: Any): pass
    def set_cost_map(self, cost_map: Any): pass

class PlanningDebugger(IDebugger):
    """
    真正的记录器
    用于开发和演示。
    """
    def __init__(self):
        # 存储格式: List[Tuple[float, float, float, float]]
        self.open_set_history: List[Tuple[float, float, float, float]] = []
        # 存储格式: List[State]
        self.expanded_nodes: List[Any] = []
        # 存储格式: List[Tuple[State, State]] 用于 RRT 的树枝
        self.edges: List[Tuple[Any, Any]] = []
        self.cost_map = None

    def record_open_set_node(self, node: Any, f: float = 0.0, h: float = 0.0):
        # 我们存下 f 和 h，虽然现在的 plotter 可能只画点，
        # 但未来你可以根据 f 值画出不同颜色的点（热力图）
        # 如果是 State 对象，取其属性，否则假设它有 x, y
        x = getattr(node, 'x', node[0] if isinstance(node, (list, tuple)) else 0)
        y = getattr(node, 'y', node[1] if isinstance(node, (list, tuple)) else 0)
        self.open_set_history.append((x, y, f, h))

    def record_current_expansion(self, node: Any):
        # 直接存储节点对象 (State 或 Node)
        self.expanded_nodes.append(node)

    def record_edge(self, start_node: Any, end_node: Any):
        """记录 RRT 的一条边"""
        self.edges.append((start_node, end_node))

    def set_cost_map(self, cost_map: Any):
        self.cost_map = cost_map