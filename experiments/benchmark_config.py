import numpy as np
import sys
import os

# Ensure src can be imported if this config is used standalone or imported from elsewhere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vehicles.config import AckermannConfig
from src.types import State
from src.collision.checker import CollisionConfig, CollisionMethod

class BenchmarkConfig:
    # --- Experiment Settings ---
    DENSITIES = [0.10, 0.15, 0.20]       # Obstacle densities to test
    NUM_TRIALS = 10                 # Number of trials per density
    RANDOM_SEED_BASE = 1000        # Base seed for reproducibility
    
    # --- Experiment Mode ---
    ENABLE_DYNAMIC_PERCEPTION = True # False = Static Planning, True = Dynamic Perception

    # --- Output Paths ---
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_DIR = os.path.dirname(_BASE_DIR)
    LOG_DIR = os.path.join(_PROJECT_DIR, "logs", "experiments_ackermann")
    # Dynamic paths will be generated in benchmark_runner.py based on timestamp

    # --- Map Parameters ---
    PHYS_WIDTH = 100.0             # meters
    PHYS_HEIGHT = 100.0            # meters
    RESOLUTION = 0.5               # meters/cell
    
    # Derived Map Settings
    MAP_WIDTH = int(PHYS_WIDTH / RESOLUTION)
    MAP_HEIGHT = int(PHYS_HEIGHT / RESOLUTION)

    # --- Start & Goal ---
    START_STATE = State(5.0, 5.0, 0.0)
    GOAL_STATE = State(90.0, 90.0, 0.0)
    
    # Area clearing around start/goal
    CLEAR_RADIUS = 4.0             # meters
    
    # Simulation Limits
    MAX_STEPS = 2000               # Increased for dense RRT paths

    # --- Map Generation (Bulldozer) ---
    EXTRA_PATHS = 6
    DEAD_ENDS = 4
    
    PLOW_CONFIG = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=35.0, 
        width=3.0,
        front_hang=1.2, 
        rear_hang=1.2, 
        safe_margin=0.5
    )

    # --- Vehicle Configuration ---
    VEHICLE_CONFIG = AckermannConfig(
        wheelbase=2.5, 
        max_steer_deg=40.0,  # Improved maneuverability
        width=2.0, 
        front_hang=1.0, 
        rear_hang=1.0,
        safe_margin=0.2
    )

    # --- Collision Checking ---
    COLLISION_CONFIG = CollisionConfig(method=CollisionMethod.POLYGON)

    # --- Algorithm Parameters ---
    
    # 1. RRT Planner
    RRT_PARAMS = {
        'step_size': 2.0,          # Reduced for finer steering in tight spaces
        'max_iterations': 10000,
        'goal_sample_rate': 0.15,  # Increased goal bias
        'goal_threshold': 2.0
    }

    # 2. RRT Smoothing
    SMOOTHER_PARAMS = {
        'max_iterations': 150
    }

    # 3. Hybrid A*
    HYBRID_ASTAR_PARAMS = {
        'xy_resolution': 0.5,
        'theta_resolution': np.deg2rad(5.0),
        'step_size': 1.5,
        'analytic_expansion_ratio': 0.2
    }
