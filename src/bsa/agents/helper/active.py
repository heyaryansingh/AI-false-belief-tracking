"""Active Verification Helper Agent.

This agent extends the Goal-Only helper.
When it detects confusion (Policy Divergence), it does NOT just intervene locally.
It triggers an ACTIVE VERIFICATION subroutine:
1. Identify best alternative location for the object.
2. Move to that location to check.
3. If found, Communicate.
"""

from .goal_only import GoalOnlyHelper
from ...common.types import Action, Observation, EpisodeStep
from typing import Optional

class ActiveVerificationHelper(GoalOnlyHelper):
    """Helper that actively verifies information when confusion is detected."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verification_mode = False
        self.target_room = None
        
    def plan_action(self, observation: Observation, episode_step: Optional[EpisodeStep] = None) -> Action:
        # Detection logic from GoalOnly
        confusion_score = self.compute_false_belief_confidence(episode_step)
        
        if self.verification_mode:
            # We are verifying. 
            # If we see the object, SAY it!
            # If we reached room and didn't see it, go back to wait?
            pass # Implementation
            
        if confusion_score > 0.7 and not self.verification_mode:
            # Trigger active verification
            self.verification_mode = True
            # Build hypothesis: "If human is in Kitchen and confused, object might be in Bedroom"
            # Return MOVE action towards Bedroom
            return Action.MOVE
            
        return super().plan_action(observation, episode_step)
