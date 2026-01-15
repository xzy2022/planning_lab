import matplotlib.pyplot as plt
import numpy as np
from src.simulation.navigator import Navigator

class SimulationVisualizer:
    def __init__(self, title="Simulation"):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.title = title
        self.initialized = False
        
        # Plot handles
        self.global_im = None
        self.local_im = None
        self.path_line = None
        self.vehicle_plot = None
        self.start_plot = None
        self.goal_plot = None

    def update(self, navigator: Navigator, pause_interval: float = 0.01):
        if not self.initialized:
            self._init_plot(navigator)
            self.initialized = True
        
        # Update Local Map Overlay
        # Mask free space (0) so we only see obstacles (1)
        local_data = navigator.local_map.data
        masked_local = np.ma.masked_where(local_data == 0, local_data)
        self.local_im.set_data(masked_local)
        
        # Update Vehicle Position
        state = navigator.current_state
        self.vehicle_plot.set_data([state.x], [state.y])
        
        # Update Navigated Path
        if navigator.navigated_path:
            px = [s.x for s in navigator.navigated_path]
            py = [s.y for s in navigator.navigated_path]
            self.path_line.set_data(px, py)
        
        # Title with step info
        self.ax.set_title(f"{self.title} | Step: {navigator.step_count} | Replans: {navigator.replan_count}")
        
        # Refresh
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause_interval) # Small pause to allow GUI update

    def update_from_state(self, state_data: dict, pause_interval: float = 0.01):
        if not self.initialized:
            # We assume init_plot has been called manually or we need a way to pass global map info
            # Ideally, init_plot should be separated or called before loop.
            pass
        
        # Unpack Data
        local_data = state_data['local_map_data']
        vehicle_x = state_data['vehicle_x']
        vehicle_y = state_data['vehicle_y']
        path_x = state_data['path_x']
        path_y = state_data['path_y']
        step = state_data.get('step', 0)
        replans = state_data.get('replans', 0)

        # Update Local Map Overlay
        masked_local = np.ma.masked_where(local_data == 0, local_data)
        self.local_im.set_data(masked_local)
        
        # Update Vehicle Position
        self.vehicle_plot.set_data([vehicle_x], [vehicle_y])
        
        # Update Navigated Path
        if path_x:
            self.path_line.set_data(path_x, path_y)
        
        # Title
        self.ax.set_title(f"{self.title} | Step: {step} | Replans: {replans}")
        
        # Refresh
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(pause_interval)

    def _init_plot(self, navigator: Navigator):
        # 1. Global Map (Background)
        grid_map = navigator.global_map
        extent = [0, grid_map.width * grid_map.resolution, 
                  0, grid_map.height * grid_map.resolution]
        
        self.global_im = self.ax.imshow(
            grid_map.data, cmap='Greys', origin='lower', extent=extent, alpha=0.3, vmin=0, vmax=1
        )
        
        # 2. Local Map (Overlay) - Initialize with empty mask
        empty_data = np.ma.masked_all(grid_map.data.shape)
        self.local_im = self.ax.imshow(
            empty_data, cmap='Reds', origin='lower', extent=extent, alpha=0.6, vmin=0, vmax=1
        )
        
        # 3. Path
        self.path_line, = self.ax.plot([], [], 'b.-', linewidth=2, markersize=4, label='Path')
        
        # 4. Vehicle
        self.vehicle_plot, = self.ax.plot([], [], 'bo', markersize=8, label='Vehicle')
        
        # 5. Start / Goal
        self.start_plot, = self.ax.plot(navigator.start.x, navigator.start.y, 'go', markersize=10, label='Start')
        self.goal_plot, = self.ax.plot(navigator.goal.x, navigator.goal.y, 'rx', markersize=10, label='Goal')
        
        self.ax.legend(loc='upper left')
        self.ax.grid(False)
