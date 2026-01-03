# tests\planning\test_a_star_planning.py
import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np

# --- 路径设置 (确保能导入 src) ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.point_mass import PointMassVehicle, PointMassConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.planners.a_star import AStarPlanner
from src.planning.heuristics.euclidean import EuclideanHeuristic
from src.visualization.debugger import PlanningDebugger
from src.planning.costs.base import CostFunction
from src.planning.costs.distance_cost import DistanceCost
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig

# --- 临时定义 DistanceCost (以防 src/planning/costs/distance_cost.py 为空) ---
# 正式代码中请使用: 
class SimpleDistanceCost(CostFunction):
    def calculate(self, current: State, next_node: State) -> float:
        return math.hypot(next_node.x - current.x, next_node.y - current.y)

def test_a_star_planning():
    print("=== 开始 A* 规划测试 ===")

    # 1. 初始化地图 (50x50, 分辨率 1.0m)
    width, height, res = 200, 200, 0.5
    grid_map = GridMap(width=width, height=height, resolution=res)
    
    # 2. 生成随机地图
    print("生成地图中...")
    config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=34.4,
        width=2.0  # 显式确认车宽，便于观察
    )
    vehicle = AckermannVehicle(config)
    obstacle_density = 0.1
    generator = MapGenerator(
        obstacle_density=obstacle_density, 
        inflation_radius_m=0.5, # 这是障碍物的膨胀，不是车身的
        num_waypoints=5,
        seed=42 
    )
    start_state = State(5.0, 5.0, 0.0)
    goal_state = State(90.0, 90.0, 0.0)
    
    # 2. 执行生成 (Generator 内部会自动处理“胖”车身逻辑)
    # extra_paths=1: 至少会有两条路通往终点
    # dead_ends=5: 生成5条乱七八糟的干扰路径
    generator.generate(grid_map, vehicle, start_state, goal_state, extra_paths=1, dead_ends=5)
    


    # 3. 配置车辆与碰撞检测
    # 质点模型
    vehicle_config = PointMassConfig(width=1.0, length=1.0, safe_margin=0.1)
    vehicle = PointMassVehicle(vehicle_config)

    

    # 碰撞检测器 (对于 Grid A*，RASTER 模式或 CIRCLE_ONLY 都可以，这里用 CIRCLE_ONLY 简单快速)
    col_config = CollisionConfig(method=CollisionMethod.CIRCLE_ONLY)
    collision_checker = CollisionChecker(col_config, vehicle, grid_map)

    # 4. 配置规划器组件 (策略模式)
    heuristic = EuclideanHeuristic()
    cost_fn = SimpleDistanceCost() # 使用上面定义的类
    
    planner = AStarPlanner(
        vehicle_model=vehicle,
        collision_checker=collision_checker,
        heuristic=heuristic,
        cost_functions=[cost_fn],
        weights=[1.0] # 距离代价权重为 1
    )

    # 5. 初始化调试器 (用于记录搜索过程)
    debugger = PlanningDebugger()

    # 6. 执行规划
    print(f"开始规划: {start_state} -> {goal_state}")
    path = planner.plan(start_state, goal_state, grid_map, debugger=debugger)

    if not path:
        print("规划失败！未找到路径。")
    else:
        print(f"规划成功！路径长度: {len(path)} 节点")

    # 7. 可视化
    visualize_result(grid_map, path, debugger, start_state, goal_state)

def visualize_result(grid_map, path, debugger, start, goal):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # A. 绘制地图背景 (转置以匹配 xy 坐标系习惯，origin='lower')
    # GridMap.data 是 (height, width)
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)

    # B. 绘制已探索节点 (Expanded Nodes) - 红色小点
    # debugger.expanded_nodes 存储的是 [(x, y), ...]
    if debugger.expanded_nodes:
        ex_x = [p[0] for p in debugger.expanded_nodes]
        ex_y = [p[1] for p in debugger.expanded_nodes]
        ax.scatter(ex_x, ex_y, c='red', s=2, alpha=0.3, label='Expanded Nodes')

    # C. 绘制 OpenSet 历史 (可选) - 绿色小点
    if debugger.open_set_history:
       print("OpenSet History:",len(debugger.open_set_history))
       op_x = [p[0] for p in debugger.open_set_history]
       op_y = [p[1] for p in debugger.open_set_history]
       ax.scatter(op_x, op_y, c='green', s=1, alpha=0.3, label='OpenSet History')

    # D. 绘制最终路径 - 蓝色实线
    if path:
        path_x = [s.x for s in path]
        path_y = [s.y for s in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2.5, label='Planned Path')
        # 画出路径点
        ax.scatter(path_x, path_y, c='blue', s=10, zorder=5)

    # E. 绘制起点和终点
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')

    ax.set_title("A* Planning with PointMass Vehicle")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3)
    
    # 确保比例尺一致
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_a_star_planning()