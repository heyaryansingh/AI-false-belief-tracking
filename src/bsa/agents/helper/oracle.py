"""Oracle Helper Agent.

This agent has God Mode (access to true state).
Used as an Upper Bound baseline.
"""

from .base import HelperAgent
from ...common.types import Action, Observation, EpisodeStep
from typing import Optional

class OracleHelper(HelperAgent):
    """God-Mode Helper."""
    
    def __init__(self, seed: Optional[int] = None):
        pass

    
    def plan_action(self, observation: Observation, episode_step: Optional[EpisodeStep] = None) -> Action:
        if episode_step is None:
            return Action.WAIT
            
        # 1. Check Divergence: True Loc vs Human Belief
        # We cheat and look at episode_step.human_belief vs true
        
        divergence = False
        for obj, loc in episode_step.human_belief_object_locations.items():
            true_loc = episode_step.true_object_locations.get(obj)
            if true_loc and loc.room_id != true_loc.room_id:
                divergence = True
                break
                
        if divergence:
            return Action.SAY # Correct them immediately
            
        return Action.WAIT # Perfect efficiency
        
    def update_belief(self, *args, **kwargs):
        pass
        
    def get_belief_state(self):
        return {"god_mode": True}
        
    def detect_false_belief(self, observation, episode_step=None) -> bool:
        if episode_step is None: return False
        # Perfect detection
        for obj, loc in episode_step.human_belief_object_locations.items():
            true_loc = episode_step.true_object_locations.get(obj)
            if true_loc and loc.room_id != true_loc.room_id:
                return True
        return False
