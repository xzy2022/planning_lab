from src.map.grid_map import GridMap
from src.map.generator import MapGenerator
from src.vehicles.ackermann import AckermannVehicle
from src.types import State

def test_feasible_map_generation():
    # 1. 准备组件
    grid_map = GridMap(100, 100, resolution=0.1)
    vehicle = AckermannVehicle(wheelbase=2.5, max_steer=0.6)
    
    start = State(5.0, 5.0, 0.0)
    goal = State(90.0, 90.0, 0.0)
    
    # 2. 调用生成器 (上帝之手)
    print("生成包含可行路径的地图中...")
    MapGenerator.generate_feasible_map(
        grid_map, vehicle, start, goal, obstacle_density=0.6
    )
    
    # 3. 后处理：适当膨胀剩余的障碍物 (可选)
    # 注意：不要膨胀得把刚才推出来的路又堵死了，所以通常先膨胀再推平，
    # 或者推平的半径要比膨胀半径大很多。
    
    # 4. 可视化 (使用之前的代码)
    import matplotlib.pyplot as plt
    plt.imshow(grid_map.data, cmap='Greys', origin='lower')
    
    # 画出起点终点
    plt.plot(start.x/0.1, start.y/0.1, 'go', label='Start')
    plt.plot(goal.x/0.1, goal.y/0.1, 'rx', label='Goal')
    plt.legend()
    plt.show()