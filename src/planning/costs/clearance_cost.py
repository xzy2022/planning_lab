# 障碍物距离惩罚 (靠近障碍物代价高)

class ObstacleClearanceCost(CostFunction):
    def __init__(self, grid_map, danger_threshold: float):
        # [关键] 把地图存下来
        self.grid_map = grid_map
        self.threshold = danger_threshold

    def calculate(self, current: State, next_node: State) -> float:
        # 查询地图（假设地图有 get_distance 方法）
        dist = self.grid_map.get_distance_to_obstacle(next_node.x, next_node.y)
        
        if dist < self.threshold:
            return 10.0 / (dist + 0.1)  # 越近代价越大
        return 0.0