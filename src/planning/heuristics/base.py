from abc import ABC, abstractmethod
from src.types import State

class Heuristic(ABC):
    @abstractmethod
    def estimate(self, current: State, goal: State) -> float:
        """统一接口：只接受当前点和目标点"""
        pass