# tests/planning/test_rrt_planning.py
import sys
import os
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# --- 路径设置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.point_mass import PointMassVehicle, PointMassConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.planners import AStarPlanner, RRTPlanner 
from src.visualization.debugger import PlanningDebugger
from src.planning.costs.base import CostFunction
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.planning.costs import DistanceCost, ClearanceCost
from src.planning.heuristics import EuclideanHeuristic, OctileHeuristic



def test_a_star_planning():
    print("=== 开始 A* 规划测试 (动画版) ===")

    # 1. 初始化地图
    width, height, res = 200, 200, 0.5
    grid_map = GridMap(width=width, height=height, resolution=res)
    
    # 2. 生成随机地图
    print("生成地图中...")
    config = AckermannConfig(wheelbase=2.5, max_steer_deg=34.4, width=2.0)
    vehicle_gen = AckermannVehicle(config)
    
    # 稍微提高一点难度或密度
    generator = MapGenerator(
        obstacle_density=0.15, 
        inflation_radius_m=0.5, 
        num_waypoints=5,
        seed=42 
    )
    start_state = State(5.0, 5.0, 0.0)
    goal_state = State(90.0, 90.0, 0.0)
    
    generator.generate(grid_map, vehicle_gen, start_state, goal_state, extra_paths=1, dead_ends=5)

    # 3. 配置规划用的车辆 (PointMass)
    vehicle_config = PointMassConfig(width=1.0, length=1.0, safe_margin=0.1)
    vehicle = PointMassVehicle(vehicle_config)

    # 4. 碰撞检测
    col_config = CollisionConfig(method=CollisionMethod.CIRCLE_ONLY)
    collision_checker = CollisionChecker(col_config, vehicle, grid_map)

    # 5. 规划器
    euclidean_heuristic = EuclideanHeuristic()
    octile_heuristic = OctileHeuristic()
    dist_cost = DistanceCost()
    clearance_cost = ClearanceCost(grid_map, risk_dist=1.0, weight_factor=5.0)
    
    rrt_planner = RRTPlanner(
        vehicle_model=vehicle,
        collision_checker=collision_checker,
        step_size=2.0,       # 每次生长 2 米
        max_iterations=5000, # 尝试 5000 次
        goal_sample_rate=0.1 # 10% 概率直接朝向终点
    )

    debugger = PlanningDebugger()

    # 6. 执行规划
    print(f"开始规划: {start_state} -> {goal_state}")
    path = rrt_planner.plan(start_state, goal_state, grid_map, debugger=debugger)

    if not path:
        print("规划失败！")
    else:
        print(f"规划成功！路径长度: {len(path)} 节点")
        print(f"一共探索了 {len(debugger.expanded_nodes)} 个节点")

    # 7. 动画可视化
    visualize_result_as_animation(grid_map, path, debugger, start_state, goal_state)

def visualize_result_as_animation(grid_map, path, debugger, start, goal):
    """
    使用 Matplotlib FuncAnimation 制作搜索过程动画
    """
    print("正在渲染动画，请稍候...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. 静态背景：地图、起点、终点
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
    
    # 2. 动态元素初始化
    # Expanded Nodes (红色) - 初始为空
    scatter_expanded = ax.scatter([], [], c='red', s=5, alpha=0.6, label='Expanded (Searching)')
    
    # 最终路径 (蓝色) - 初始不显示，最后显示
    line_path, = ax.plot([], [], 'b-', linewidth=2.5, label='Planned Path', alpha=0.0)
    
    # 设置标题和标签
    title_text = ax.set_title("A* Search Progress: Frame 0")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_aspect('equal')

    # --- 动画数据准备 ---
    # 转换 list 为 numpy array 以提高性能
    expanded_data = np.array(debugger.expanded_nodes) if debugger.expanded_nodes else np.empty((0, 2))
    
    total_nodes = len(expanded_data)
    
    # [关键技巧] 计算步长
    # 如果节点太多（比如1万个），我们不想生成1万帧，那样太慢了。
    # 我们希望动画大概在 5-10 秒内播完。假设 30fps，也就是 300 帧左右。
    target_frames = 300
    if total_nodes > 0:
        steps_per_frame = max(1, total_nodes // target_frames)
    else:
        steps_per_frame = 1
        
    print(f"总节点数: {total_nodes}, 每帧新增显示: {steps_per_frame} 个节点")

    def init():
        scatter_expanded.set_offsets(np.empty((0, 2)))
        line_path.set_data([], [])
        line_path.set_alpha(0.0)
        return scatter_expanded, line_path, title_text

    def update(frame):
        # frame 是当前的帧数索引
        # 计算当前应该显示到第几个节点
        current_idx = min(frame * steps_per_frame, total_nodes)
        
        # 1. 更新红色探索点
        if current_idx > 0:
            # 取前 current_idx 个点
            current_data = expanded_data[:current_idx]
            scatter_expanded.set_offsets(current_data)
        
        # 更新标题
        title_text.set_text(f"A* Searching... Nodes: {current_idx}/{total_nodes}")

        # 2. 如果搜索结束（最后几帧），显示最终路径
        if current_idx >= total_nodes and path:
            path_x = [s.x for s in path]
            path_y = [s.y for s in path]
            line_path.set_data(path_x, path_y)
            line_path.set_alpha(1.0) # 显示路径
            title_text.set_text("Search Compelted! Path Found.")

        return scatter_expanded, line_path, title_text

    # 创建动画
    # frames: 总帧数 (为了多停顿一会儿让大家看清结果，多加 20 帧)
    total_frames = (total_nodes // steps_per_frame) + 30
    
    ani = FuncAnimation(
        fig, 
        update, 
        frames=total_frames, 
        init_func=init, 
        interval=20, # 每帧间隔 20ms (约 50fps)
        blit=True,   # 开启 blit 优化绘图性能
        repeat=False # 播放一次后停止
    )
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_a_star_planning()