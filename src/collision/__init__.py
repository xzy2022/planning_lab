# src/collision/__init__.py

from .config import CollisionConfig, CollisionMethod
from .checker import CollisionChecker
from .footprint import FootprintModel
# geometry 通常作为底层库，不需要直接暴露到顶层，除非你经常单独使用它

__all__ = [
    "CollisionConfig", 
    "CollisionMethod", 
    "CollisionChecker",
    "FootprintModel"
]