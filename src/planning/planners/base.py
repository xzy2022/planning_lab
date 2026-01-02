# src/planning/planners/base.py
from abc import ABC, abstractmethod
from typing import List, Optional
from src.types import State
from src.map.grid_map import GridMap
from src.visualization.debugger import IDebugger

class PlannerBase(ABC):
    """
    所有路径规划器的抽象基类
    """
    
    @abstractmethod
    def plan(self, 
             start: State, 
             goal: State, 
             grid_map: GridMap,
             debugger: Optional[IDebugger] = None) -> List[State]:
        """
        执行路径规划
        :param start: 起点状态
        :param goal: 目标状态
        :param grid_map: 环境地图
        :param debugger: 调试器钩子 (用于可视化搜索过程)
        :return: 路径点列表 (如果失败返回空列表)
        """
        pass