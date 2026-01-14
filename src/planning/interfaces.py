from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class IPlannerObserver(ABC):
    """
    规划器观察者接口
    用于解耦规划算法与 记录/调试/可视化 逻辑。
    支持三种模式：
    1. Efficient: 空实现，无开销
    2. Experiment: 记录关键数据用于可视化
    3. Debug: 详细日志记录用于问题排查
    """
    
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
        """记录图/树的一条边"""
        pass
    
    @abstractmethod
    def set_map_info(self, map_info: Any): 
        """设置地图信息 (用于可视化背景等)"""
        pass
        
    @abstractmethod
    def log(self, message: str, level: str = 'INFO', payload: Optional[Dict] = None):
        """
        结构化日志记录
        :param message: 日志消息
        :param level: 日志级别 'INFO', 'WARN', 'ERROR', 'DEBUG'
        :param payload: 额外的结构化数据 (如状态详情、配置参数等)
        """
        pass
