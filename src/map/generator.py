# src/map/generator.py
import numpy as np
import math
import random
from src.map.grid_map import GridMap
from src.types import State
from src.vehicles.base import VehicleBase

class MapGenerator:
    """
    地图生成器 (实例版)
    将配置参数与生成逻辑绑定，方便管理复杂参数。
    """
    
    def __init__(
        self, 
        obstacle_density: float = 0.1, 
        inflation_radius_m: float = 0.2,  # [改进] 这里直接传米
        num_waypoints: int = 5,
        max_steps: int = 3000,
        seed: int = None
    ):
        self.density = obstacle_density
        self.inflation_radius_m = inflation_radius_m
        self.num_waypoints = num_waypoints
        self.max_steps = max_steps
        self.seed = seed
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

    def generate(self, grid_map: GridMap, vehicle: VehicleBase, start: State, goal: State):
        """
        主入口：生成一张完整的可行地图
        """
        # 1. 随机障碍底图
        self._generate_random_obstacles(grid_map)
        
        # 2. 障碍物膨胀 (物理距离 -> 栅格)
        # 自动计算 grid 数量，向上取整保证至少膨胀 1 格（如果半径>0）
        if self.inflation_radius_m > 0:
            r_grids = int(math.ceil(self.inflation_radius_m / grid_map.resolution))
            self._inflate_obstacles(grid_map, r_grids)
            
        # 3. 清除起终点
        # 保证起点终点周围有足够的回旋余地 (例如 3米)
        self._clear_area_m(grid_map, start.x, start.y, radius_m=3.0)
        self._clear_area_m(grid_map, goal.x, goal.y, radius_m=3.0)
        
        # 4. 生成路点并推土
        self._carve_path(grid_map, vehicle, start, goal)

    def _generate_random_obstacles(self, grid_map: GridMap):
        width = grid_map.width
        height = grid_map.height
        
        random_mask = np.random.rand(height, width) < self.density
        grid_map.data[random_mask] = 1
        
        # 围墙
        grid_map.data[0, :] = 1
        grid_map.data[-1, :] = 1
        grid_map.data[:, 0] = 1
        grid_map.data[:, -1] = 1

    def _inflate_obstacles(self, grid_map: GridMap, radius_grids: int):
        """具体的膨胀逻辑 (操作 Grid)"""
        if radius_grids <= 0: return

        data = grid_map.data
        rows, cols = data.shape
        obs_y, obs_x = np.where(data == 1)
        
        inflated_grid = data.copy()
        
        for y, x in zip(obs_y, obs_x):
            y_min = max(0, y - radius_grids)
            y_max = min(rows, y + radius_grids + 1)
            x_min = max(0, x - radius_grids)
            x_max = min(cols, x + radius_grids + 1)
            inflated_grid[y_min:y_max, x_min:x_max] = 1
            
        grid_map.data[:] = inflated_grid[:]

    def _carve_path(self, grid_map: GridMap, vehicle: VehicleBase, start: State, goal: State):
        """推土机逻辑"""
        # 生成路点
        waypoints = []
        margin = min(grid_map.width, grid_map.height) * grid_map.resolution * 0.1
        min_x, max_x = margin, grid_map.width * grid_map.resolution - margin
        min_y, max_y = margin, grid_map.height * grid_map.resolution - margin

        for _ in range(self.num_waypoints):
            rx = random.uniform(min_x, max_x)
            ry = random.uniform(min_y, max_y)
            waypoints.append(State(rx, ry, 0.0))
        waypoints.append(goal)
        
        # 运行车辆
        current_state = start
        dt = 0.5
        max_steer = getattr(vehicle.config, 'max_steer', 0.6)
        target_idx = 0
        step_count = 0

        while step_count < self.max_steps and target_idx < len(waypoints):
            target = waypoints[target_idx]
            dx = target.x - current_state.x
            dy = target.y - current_state.y
            
            if math.hypot(dx, dy) < 4.0:
                target_idx += 1
                if target_idx >= len(waypoints):
                    print("Generator: Path carved successfully.")
                    break
                continue

            # 简单的 P 控制
            target_yaw = math.atan2(dy, dx)
            diff_yaw = target_yaw - current_state.theta_rad
            diff_yaw = (diff_yaw + math.pi) % (2 * math.pi) - math.pi
            steer = max(min(diff_yaw, max_steer), -max_steer)
            
            next_state = vehicle.kinematic_propagate(current_state, (1.0, steer), dt)
            
            # 清除障碍 (物理尺寸)
            bx, by, b_radius = vehicle.get_bounding_circle(next_state)
            self._clear_area_m(grid_map, bx, by, radius_m=b_radius * 1.2)
            
            current_state = next_state
            step_count += 1

    def _clear_area_m(self, grid_map: GridMap, center_x: float, center_y: float, radius_m: float):
        """[封装] 统一处理物理距离到栅格的转换"""
        cx_idx, cy_idx = grid_map.world_to_grid(center_x, center_y)
        r_grids = int(math.ceil(radius_m / grid_map.resolution))
        
        # 边界检查 + 圆形判断
        for y in range(cy_idx - r_grids, cy_idx + r_grids + 1):
            for x in range(cx_idx - r_grids, cx_idx + r_grids + 1):
                if 0 <= x < grid_map.width and 0 <= y < grid_map.height:
                    if (x - cx_idx)**2 + (y - cy_idx)**2 <= r_grids**2:
                        grid_map.data[y, x] = 0