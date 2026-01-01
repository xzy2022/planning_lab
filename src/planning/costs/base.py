class CostFunction(ABC):
    @abstractmethod
    def calculate(self, current: State, next_node: State) -> float:
        pass