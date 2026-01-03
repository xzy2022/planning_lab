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
    地图生成器 (增强版)
    支持多路径冗余和死胡同陷阱生成
    """
    
    def __init__(
        self, 
        obstacle_density: float = 0.1, 
        inflation_radius_m: float = 0.2, 
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

    def generate(self, grid_map: GridMap, vehicle: VehicleBase, start: State, goal: State, 
                 extra_paths: int = 1,  # [新增] 额外生成几条通往终点的路
                 dead_ends: int = 3):   # [新增] 生成几条死胡同/干扰路
        
        # 1. 随机障碍底图
        self._generate_random_obstacles(grid_map)
        
        # 2. 障碍物膨胀
        if self.inflation_radius_m > 0:
            r_grids = int(math.ceil(self.inflation_radius_m / grid_map.resolution))
            self._inflate_obstacles(grid_map, r_grids)

        # 3. 准备“推土机”模型
        plow_vehicle = copy.deepcopy(vehicle)
        # 增加推土机尺寸作为安全余量 (e.g., 20%)
        if hasattr(plow_vehicle.config, 'width'):
            plow_vehicle.config.width *= 1.2 
        if hasattr(plow_vehicle.config, 'length'):
            plow_vehicle.config.length *= 1.2
        plow_vehicle.config.__post_init__()
        
        footprint_model = FootprintModel(plow_vehicle, grid_map.resolution)

        # 4. 清除关键点 (起点和终点)
        self._clear_with_footprint(grid_map, footprint_model, start)
        self._clear_with_footprint(grid_map, footprint_model, goal)
        
        # --- [核心逻辑优化] ---
        
        # 5.1 生成主路径 (Main Path)
        print("Carving Main Path...")
        self._carve_path_with_footprint(grid_map, vehicle, footprint_model, start, goal)

        # 5.2 生成冗余路径 (Alternative Paths)
        # 同样的起终点，但因为内部 waypoints 是随机的，会走出不同的轨迹
        for i in range(extra_paths):
            print(f"Carving Alternative Path {i+1}...")
            self._carve_path_with_footprint(grid_map, vehicle, footprint_model, start, goal)

        # 5.3 生成死胡同/陷阱 (Dead Ends)
        # 随机取两点进行连接。如果运气好，它会接上主路形成分支；如果接不上，就是干扰项。
        for i in range(dead_ends):
            print(f"Carving Dead End {i+1}...")
            # 随机生成假起点和假终点
            fake_start = self._get_random_state(grid_map)
            fake_goal = self._get_random_state(grid_map)
            
            # 清除假起终点周围的障碍，确保推土机能放下
            self._clear_with_footprint(grid_map, footprint_model, fake_start)
            self._clear_with_footprint(grid_map, footprint_model, fake_goal)
            
            self._carve_path_with_footprint(grid_map, vehicle, footprint_model, fake_start, fake_goal)

    def _get_random_state(self, grid_map: GridMap) -> State:
        """生成地图范围内的随机状态"""
        margin = 2.0 # 留一点边距
        phys_w = grid_map.width * grid_map.resolution
        phys_h = grid_map.height * grid_map.resolution
        
        rx = random.uniform(margin, phys_w - margin)
        ry = random.uniform(margin, phys_h - margin)
        rtheta = random.uniform(-math.pi, math.pi)
        return State(rx, ry, rtheta)

    def _generate_random_obstacles(self, grid_map: GridMap):
        # ... (保持原代码不变) ...
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
        # ... (保持原代码不变) ...
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


    def _carve_path_with_footprint(self, grid_map: GridMap, vehicle: VehicleBase, footprint_model: FootprintModel, start: State, goal: State):
        waypoints = []
        margin = min(grid_map.width, grid_map.height) * grid_map.resolution * 0.1
        min_x, max_x = margin, grid_map.width * grid_map.resolution - margin
        min_y, max_y = margin, grid_map.height * grid_map.resolution - margin

        # 1. 生成随机路点
        for _ in range(self.num_waypoints):
            rx = random.uniform(min_x, max_x)
            ry = random.uniform(min_y, max_y)
            rtheta_rad = random.uniform(-math.pi, math.pi)
            waypoints.append(State(rx, ry, rtheta_rad))
        waypoints.append(goal)
        
        current_state = start
        
        # [修改点] 定义单步最大生长距离 (类似 RRT 的 step_size)
        # 这个值决定了“推土机”清理障碍的精细程度。
        # 如果太大 (如 10m)，PointMass 会直接跳跃，导致中间的障碍没被清除。
        # 建议设为 0.5m - 1.0m 左右，确保 footprint 能够覆盖路径。
        step_size = 1.0 
        
        target_idx = 0
        total_steps = 0
        
        # 2. 逐个路点逼近
        while total_steps < self.max_steps and target_idx < len(waypoints):
            target = waypoints[target_idx]
            
            # 计算距离
            dist = math.hypot(target.x - current_state.x, target.y - current_state.y)
            
            # 如果距离很近，切换到下一个路点
            if dist < step_size:
                target_idx += 1
                continue
            
            # [核心解耦]：委托 Vehicle 自己决定如何向 Target 移动一步
            # - Ackermann: 会计算 steer，并返回一段细腻的轨迹 (list[State])
            # - PointMass: 会计算直线位移，并返回终点
            next_state, trajectory = vehicle.propagate_towards(
                start=current_state, 
                target=target, 
                max_dist=step_size
            )
            
            # 3. 沿途清除障碍 (Footprint 推土机)
            # 无论 trajectory 是包含多个密集的点 (Ackermann) 还是仅包含端点 (PointMass)，
            # 只要 step_size 足够小 (小于车长)，就能保证路径被连续清除。
            for s in trajectory:
                self._clear_with_footprint(grid_map, footprint_model, s)
            
            # 更新状态
            current_state = next_state
            total_steps += 1

    def _clear_with_footprint(self, grid_map: GridMap, model: FootprintModel, state: State):
        indices = model.get_occupied_indices(state)
        valid_mask = (indices[:, 0] >= 0) & (indices[:, 0] < grid_map.width) & \
                     (indices[:, 1] >= 0) & (indices[:, 1] < grid_map.height)
        valid_indices = indices[valid_mask]
        if len(valid_indices) > 0:
            grid_map.data[valid_indices[:, 1], valid_indices[:, 0]] = 0
