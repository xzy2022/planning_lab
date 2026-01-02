# src/types.py
from dataclasses import dataclass

@dataclass
class State:
    """
    统一的车辆状态定义
    """
    x: float             # [m]
    y: float             # [m]
    theta_rad: float     # [rad] 注意：为了明确单位，建议保留 _rad 后缀


@dataclass
class Node:
    """搜索树节点"""
    state: State
    cost: float
    parent_index: int

    # 为了方便访问 x, y (A* 里面常用 node.x)
    @property
    def x(self): return self.state.x
    
    @property
    def y(self): return self.state.y