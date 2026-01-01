# 绘图逻辑 (Matplotlib/PyQt)

import matplotlib.pyplot as plt
from src.visualization.debugger import PlanningDebugger

class Visualizer:
    def __init__(self, map_obj):
        self.map = map_obj
        self.fig, self.ax = plt.subplots()

    def animate(self, debugger: PlanningDebugger):
        """
        根据 debugger 里的历史数据生成动画
        """
        # 1. 画静态底图
        self.ax.imshow(self.map.data, cmap='gray')
        
        # 2. 画 CostMap (如果有) 
        if debugger.cost_map is not None:
            self.ax.imshow(debugger.cost_map, alpha=0.5, cmap='jet')

        # 3. 动态画搜索过程
        # 这里可以使用 matplotlib 的 FuncAnimation
        # 读取 debugger.expanded_nodes 列表
        x_vals = [n[0] for n in debugger.expanded_nodes]
        y_vals = [n[1] for n in debugger.expanded_nodes]
        
        self.ax.scatter(x_vals, y_vals, c='red', s=2, label='Explored')
        plt.show()