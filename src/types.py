# [关键] 全局通用数据结构 (原 interfaces.py 的一部分)

# src/types.py
from dataclasses import dataclass

@dataclass
class Pose:
    """全局通用的位姿定义"""
    x: float
    y: float
    theta: float

@dataclass
class Node:
    """搜索树节点"""
    pose: Pose
    cost: float
    parent_index: int