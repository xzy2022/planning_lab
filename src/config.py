# [关键] 全局配置定义

# src/config.py
from dataclasses import dataclass

@dataclass
class GlobalConfig:
    map_resolution: float = 0.1
    max_iterations: int = 10000
    debug_mode: bool = False