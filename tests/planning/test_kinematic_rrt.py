# tests/planning/test_kinematic_rrt.py
import sys
import os
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.patches import Polygon

# --- 路径设置 ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.vehicles.config import AckermannConfig
from src.types import State
from src.collision.checker import CollisionChecker, CollisionConfig, CollisionMethod
from src.planning.planners import RRTPlanner 
from src.visualization.debugger import PlanningDebugger

def test_kinematic_rrt_planning():
    print("=== 开始阿克曼运动学 RRT 规划测试 ===")

    # 1. 初始化地图
    width, height, res = 200, 200, 0.5
    grid_map = GridMap(width=width, height=height, resolution=res)
    
    # 2. 配置阿克曼车辆
    # 轴距 2.5m, 最大转向角 35度
    vehicle_config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=35.0, 
        width=2.0, 
        front_hang=0.9, 
        rear_hang=0.9,
        safe_margin=0.2
    )
    vehicle = AckermannVehicle(vehicle_config)
    
    # 3. 准备推土机 (Bulldozer) 用于生成地图，确保路径充裕
    plow_config = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=35.0, 
        width=2.5,       # 增加宽度 (原 2.0)
        front_hang=1.2,  # 增加前悬 (原 0.9)
        rear_hang=1.2,   # 增加后悬 (原 0.9)
        safe_margin=0.5  # 增加安全边际
    )
    plow_vehicle = AckermannVehicle(plow_config)

    print("生成地图中... (使用推土机模式)")
    generator = MapGenerator(
        obstacle_density=0.20, 
        inflation_radius_m=0.1, 
        num_waypoints=3,       # 增加路点使地图更复杂一点
        seed=1234 
    )
    start_state = State(10.0, 10.0, 0.0)
    goal_state = State(90.0, 90.0, 0.0)
    
    generator.generate(grid_map, plow_vehicle, start_state, goal_state, extra_paths=4, dead_ends=2)

    # 4. 碰撞检测
    col_config = CollisionConfig(method=CollisionMethod.POLYGON)
    collision_checker = CollisionChecker(col_config, vehicle, grid_map)

    # 检查起点是否碰撞
    if collision_checker.check(vehicle, start_state, grid_map):
        print("警告: 起点处于碰撞状态！正在强制清除起点周围区域...")
        generator._clear_rectangular_area(grid_map, start_state, 4.0)
        if collision_checker.check(vehicle, start_state, grid_map):
            print("错误: 即使清除后起点仍碰撞。")

    # 5. 规划器
    rrt_planner = RRTPlanner(
        vehicle_model=vehicle,
        collision_checker=collision_checker,
        step_size=3.0,       # 减小步长以适应更复杂的地图
        max_iterations=50000,# 增加迭代次数
        goal_sample_rate=0.2, 
        goal_threshold=5.0   # 略微增大目标阈值，后续可以通过局部优化接入
    )

    debugger = PlanningDebugger()

    # 6. 执行规划
    print(f"开始规划: {start_state} -> {goal_state}")
    
    # 简单的进度打印
    import time
    start_time = time.time()
    
    # 我们拦截一下 RRTPlanner 的 plan，或者干脆加一个装饰
    path = rrt_planner.plan(start_state, goal_state, grid_map, debugger=debugger)
    
    duration = time.time() - start_time
    print(f"规划耗时: {duration:.2f}s, 树节点数: {len(rrt_planner.node_list)}")

    if not path:
        print("规划失败！")
    else:
        print(f"规划成功！路径点数: {len(path)}")
        print(f"一共探索了 {len(debugger.expanded_nodes)} 个节点")

    # 7. 可视化
    visualize_result(grid_map, path, debugger, start_state, goal_state, vehicle)

def visualize_result(grid_map, path, debugger, start, goal, vehicle):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. 静态背景：地图、起点、终点
    ax.imshow(grid_map.data, cmap='Greys', origin='lower', 
              extent=[0, grid_map.width * grid_map.resolution, 
                      0, grid_map.height * grid_map.resolution],
              alpha=0.5)
    
    ax.plot(start.x, start.y, 'go', markersize=10, label='Start')
    ax.plot(goal.x, goal.y, 'rx', markersize=10, label='Goal')
    
    # 2. 绘制 RRT 树枝 (由 edges 记录)
    if hasattr(debugger, 'edges'):
        for edge in debugger.edges:
            s1, s2 = edge
            # 对于阿克曼 RRT，简单的直线连接不准确，但在探索阶段可以作为示意
            # 更准确的是画出 propagate_towards 生成的 trajectory
            ax.plot([s1.x, s2.x], [s1.y, s2.y], 'r-', linewidth=0.5, alpha=0.3)

    # 3. 绘制最终路径
    if path:
        path_x = [s.x for s in path]
        path_y = [s.y for s in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
        
        # 4. 绘制路径上的车辆 Footprint (抽样绘制)
        sample_interval = max(1, len(path) // 20)
        for i in range(0, len(path), sample_interval):
            state = path[i]
            poly = vehicle.get_visualization_polygon(state)
            patch = Polygon(poly, closed=True, fill=False, edgecolor='blue', alpha=0.5)
            ax.add_patch(patch)
            
            # 画一个车头方向的小箭头
            ax.arrow(state.x, state.y, 1.5 * math.cos(state.theta_rad), 1.5 * math.sin(state.theta_rad),
                     head_width=0.5, head_length=0.8, fc='blue', ec='blue')

    ax.set_title("Kinematic RRT Planning (Ackermann)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), "kinematic_rrt_result.png")
    plt.savefig(save_path)
    print(f"结果已保存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    test_kinematic_rrt_planning()
