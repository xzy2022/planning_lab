# src/visualization/debugger.py
from typing import Any

# Import new interfaces and implementations
from src.planning.interfaces import IPlannerObserver
from src.visualization.observers import EfficientObserver, ExperimentObserver, DebugObserver

class IDebugger(IPlannerObserver):
    """
    [Deprecation Warning]
    Legacy interface kept for backward compatibility. 
    New code should use `IPlannerObserver` directly.
    """
    def set_cost_map(self, cost_map: Any):
        """Alias for set_map_info to maintain backward compatibility"""
        self.set_map_info(cost_map)

class NoOpDebugger(EfficientObserver, IDebugger):
    """Legacy alias for EfficientObserver"""
    pass

class PlanningDebugger(ExperimentObserver, IDebugger):
    """Legacy alias for ExperimentObserver with compat methods"""
    
    def set_cost_map(self, cost_map: Any):
        self.set_map_info(cost_map)
        
    # ExperimentObserver already implements record_open_set_node, record_current_expansion, record_edge
    # We just ensure it satisfies IDebugger which now inherits IPlannerObserver