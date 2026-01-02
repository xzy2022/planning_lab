# src/map/generator.py
import numpy as np
import math
import random
from src.map.grid_map import GridMap
from src.types import State
from src.vehicles.base import VehicleBase
from src.collision.footprint import FootprintModel
import copy

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
        # 1. 随机障碍底图 (保持不变)
        self._generate_random_obstacles(grid_map)
        
        # 2. 障碍物膨胀 (保持不变)
        if self.inflation_radius_m > 0:
            r_grids = int(math.ceil(self.inflation_radius_m / grid_map.resolution))
            self._inflate_obstacles(grid_map, r_grids)

        # --- [关键修改] 3. 准备“推土机”模型 ---
        # 我们不直接用传入的 vehicle，而是创建一个“更胖”的版本，作为安全余量
        plow_vehicle = copy.deepcopy(vehicle)
        
        # 策略：简单粗暴地增加物理尺寸
        # 比如：让推土机的宽度和长度都增加 20% 或者 固定增加 0.5米
        # 注意：这里需要根据具体的 Config 类型来处理，或者直接修改 vehicle.config.safe_margin
        # 但修改 geometric 尺寸效果最好
        if hasattr(plow_vehicle.config, 'width'):
            plow_vehicle.config.width *= 1.2 
        if hasattr(plow_vehicle.config, 'length'):
            plow_vehicle.config.length *= 1.2
            
        # 重新触发 post_init 以更新几何参数 (关键!)
        plow_vehicle.config.__post_init__()
        
        # 初始化 Footprint 模型 (计算量很小，只需做一次)
        footprint_model = FootprintModel(plow_vehicle, grid_map.resolution)

        # 4. 清除起终点 (使用 Footprint 清除)
        self._clear_with_footprint(grid_map, footprint_model, start)
        self._clear_with_footprint(grid_map, footprint_model, goal)
        
        # 5. 生成路点并推土
        self._carve_path_with_footprint(grid_map, vehicle, footprint_model, start, goal)

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

    def _carve_path_with_footprint(self, grid_map: GridMap, vehicle: VehicleBase,footprint_model:FootprintModel, start: State, goal: State):
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
            self._clear_with_footprint(grid_map, footprint_model, next_state)
            
            current_state = next_state
            step_count += 1

    def _clear_with_footprint(self, grid_map: GridMap, model: FootprintModel, state: State):
        """利用 Numpy 高速清除"""
        # 获取要清除的所有索引 (N, 2)
        indices = model.get_occupied_indices(state)
        
        # 过滤越界索引
        valid_mask = (indices[:, 0] >= 0) & (indices[:, 0] < grid_map.width) & \
                     (indices[:, 1] >= 0) & (indices[:, 1] < grid_map.height)
        valid_indices = indices[valid_mask]
        
        if len(valid_indices) > 0:
            # [极速操作] 直接赋值 0
            grid_map.data[valid_indices[:, 1], valid_indices[:, 0]] = 0