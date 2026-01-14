import logging
import time
import os
from typing import Any, List, Tuple, Dict, Optional
from src.planning.interfaces import IPlannerObserver

class EfficientObserver(IPlannerObserver):
    """
    高效运行模式
    除了必要的流程不额外进行信息记录。
    相当于 NoOp。
    """
    def record_open_set_node(self, node: Any, f: float = 0.0, h: float = 0.0): pass
    def record_current_expansion(self, node: Any): pass
    def record_edge(self, start_node: Any, end_node: Any): pass
    def set_map_info(self, map_info: Any): pass
    def log(self, message: str, level: str = 'INFO', payload: Optional[Dict] = None):
        # 仅在 ERROR 级别打印，或者完全静默
        if level == 'ERROR':
            print(f"[ERROR] {message}")


class ExperimentObserver(IPlannerObserver):
    """
    实验模式
    记录开集、闭集、拓展节点等关键算法执行内容。
    这些信息主要用于算法的比较和可视化 (Replay)。
    """
    def __init__(self):
        # 存储格式: List[Tuple[x, y, f, h]]
        self.open_set_history: List[Tuple[float, float, float, float]] = []
        # 存储格式: List[Any] (通常是 State)
        self.expanded_nodes: List[Any] = []
        # 存储格式: List[Tuple[start, end]]
        self.edges: List[Tuple[Any, Any]] = []
        self.map_info = None

    def record_open_set_node(self, node: Any, f: float = 0.0, h: float = 0.0):
        # 尝试提取 x, y
        x = getattr(node, 'x', node[0] if isinstance(node, (list, tuple)) else 0)
        y = getattr(node, 'y', node[1] if isinstance(node, (list, tuple)) else 0)
        self.open_set_history.append((x, y, f, h))

    def record_current_expansion(self, node: Any):
        self.expanded_nodes.append(node)

    def record_edge(self, start_node: Any, end_node: Any):
        self.edges.append((start_node, end_node))

    def set_map_info(self, map_info: Any):
        self.map_info = map_info
        
    def log(self, message: str, level: str = 'INFO', payload: Optional[Dict] = None):
        # 实验模式通常只关心结果和可视化，控制台输出保持简洁
        # 可以在这里做一些轻量级的打印
        pass


class DebugObserver(IPlannerObserver):
    """
    Debug 模式
    用于详细分析一次实验为什么效果不好甚至失败。
    将详细日志写入文件，同时保留可视化数据以便对照。
    """
    def __init__(self, log_dir: str = "logs/planning_debug"):
        # 复用 ExperimentObserver 的存储，以便 Debug 时也能画图
        self.viz_observer = ExperimentObserver()
        
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 配置 Logger
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"plan_debug_{timestamp}.log")
        
        self.logger = logging.getLogger(f"PlannerDebug_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        
        # 避免添加重复 Handler
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            # 同时输出到控制台 (可选)
            # ch = logging.StreamHandler()
            # ch.setLevel(logging.INFO)
            # self.logger.addHandler(ch)
            
        self.logger.info("=== Debug Session Started ===")

    def record_open_set_node(self, node: Any, f: float = 0.0, h: float = 0.0):
        self.viz_observer.record_open_set_node(node, f, h)
        # 可以在这里记录更详细的信息到 log
        # self.logger.debug(f"OpenSet Push: {node} f={f:.2f} h={h:.2f}")

    def record_current_expansion(self, node: Any):
        self.viz_observer.record_current_expansion(node)
        self.logger.debug(f"Expanding: {node}")

    def record_edge(self, start_node: Any, end_node: Any):
        self.viz_observer.record_edge(start_node, end_node)
        
    def set_map_info(self, map_info: Any):
        self.viz_observer.set_map_info(map_info)
        self.logger.info(f"Map Info set: {map_info}")

    def log(self, message: str, level: str = 'INFO', payload: Optional[Dict] = None):
        if payload:
            message = f"{message} | Payload: {payload}"
            
        if level == 'DEBUG':
            self.logger.debug(message)
        elif level == 'WARN':
            self.logger.warning(message)
        elif level == 'ERROR':
            self.logger.error(message)
        else:
            self.logger.info(message)
            
    # Proxy properties for ExperimentObserver compatibility if needed by external tools
    @property
    def expanded_nodes(self): return self.viz_observer.expanded_nodes
    @property
    def open_set_history(self): return self.viz_observer.open_set_history
    @property
    def edges(self): return self.viz_observer.edges
    @property
    def cost_map(self): return self.viz_observer.map_info 
