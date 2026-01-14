import pytest
import os
import glob
from src.types import State
from src.map.grid_map import GridMap
from src.visualization.observers import EfficientObserver, ExperimentObserver, DebugObserver
from src.planning.planners.rrt import RRTPlanner, RRTNode
from src.vehicles.ackermann import AckermannVehicle
from src.collision import CollisionChecker

class MockCollisionChecker(CollisionChecker):
    def __init__(self):
        pass
    def check(self, vehicle, state, grid_map):
        return False

class MockVehicle(AckermannVehicle):
    def __init__(self):
        pass
    def propagate_towards(self, start, target, max_dist):
        return target, [start, target]
    def normalize_angle(self, angle):
        return angle

@pytest.fixture
def planner_setup():
    grid_map = GridMap(width=20, height=20, resolution=1.0)
    vehicle = MockVehicle()
    collision_checker = MockCollisionChecker()
    # Mock specific config for vehicle if needed, but RRT mainly calls propagate_towards
    
    planner = RRTPlanner(
        vehicle_model=vehicle,
        collision_checker=collision_checker,
        step_size=5.0,
        max_iterations=10
    )
    start = State(0, 0, 0)
    goal = State(10, 10, 0)
    return planner, start, goal, grid_map

def test_efficient_mode(planner_setup):
    planner, start, goal, grid_map = planner_setup
    observer = EfficientObserver()
    
    # Run plan
    path = planner.plan(start, goal, grid_map, debugger=observer)
    
    # EfficientObserver should not record anything
    # We can't really test "not recording" internal state easily unless we inspect the object 
    # but we can ensure it runs without error and has no public attributes like 'expanded_nodes' from proper observers
    assert not hasattr(observer, 'expanded_nodes')
    assert not hasattr(observer, 'open_set_history')

def test_experiment_mode(planner_setup):
    planner, start, goal, grid_map = planner_setup
    observer = ExperimentObserver()
    
    path = planner.plan(start, goal, grid_map, debugger=observer)
    
    # Validation: should have recorded expanded nodes
    assert hasattr(observer, 'expanded_nodes')
    assert len(observer.expanded_nodes) > 0
    assert hasattr(observer, 'edges')
    assert len(observer.edges) > 0 # RRT produces edges

def test_debug_mode(planner_setup):
    planner, start, goal, grid_map = planner_setup
    
    # Clean up old logs
    log_dir = "logs/test_planning_debug"
    if os.path.exists(log_dir):
        files = glob.glob(os.path.join(log_dir, "*"))
        for f in files:
            try:
                os.remove(f)
            except:
                pass
                
    observer = DebugObserver(log_dir=log_dir)
    
    path = planner.plan(start, goal, grid_map, debugger=observer)
    
    # 1. Compatible with Experiment Mode (can retrieve expanded_nodes)
    assert len(observer.expanded_nodes) > 0
    
    # 2. Check Log File
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    assert len(log_files) > 0
    
    with open(log_files[0], 'r', encoding='utf-8') as f:
        content = f.read()
        assert "Step" in content or "Start planning" in content
