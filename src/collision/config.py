# src/collision/config.py
from enum import Enum
from dataclasses import dataclass

class CollisionMethod(Enum):
    # 仅使用外接圆检测 (最快，但非常保守，无法穿过狭窄区域)
    CIRCLE_ONLY = 0
    
    # 多圆覆盖检测 (速度快，精度较高，适合 Ackermann)
    MULTI_CIRCLE = 1
    
    # 多边形 SAT 检测 (最精确，计算量大，适合事后验证或高精度场景)
    POLYGON = 2

    # 离散栅格检测 
    RASTER = 3

@dataclass
class CollisionConfig:
    method: CollisionMethod = CollisionMethod.MULTI_CIRCLE
    # 如果需要在检测层额外增加膨胀 (除了车辆自身的 safe_margin)，可以在此配置
    # 但根据你的描述，车辆模型内部已经处理了 safe_margin，这里设为 0 即可
    extra_inflation: float = 0.0