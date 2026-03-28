"""Communication Helper Agent.

This agent breaks the Ignorance Constraint by asking.
"I don't know where the keys are. Do you?"
"""

from .goal_only import GoalOnlyHelper
from ...common.types import Action, Observation, EpisodeStep
import numpy as np
from typing import Optional

class CommunicationHelper(GoalOnlyHelper):
    """Helper that uses communication to resolve uncertainty."""
    
    def plan_action(self, observation: Observation, episode_step: Optional[EpisodeStep] = None) -> Action:
        # 1. Detect confusion (using efficiency proxy)
        confusion_score = self.compute_false_belief_confidence(episode_step)
        
        # 2. If confused, ASK
        if confusion_score > 0.6:
            # In a real system, we would get an answer.
            # In simulation, we assume 'SAY' represents asking/clarifying.
            # If we ASK, we effectively resolve the false belief (simulate success).
            return Action.SAY
            
        return super().plan_action(observation, episode_step)
