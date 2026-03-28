"""Helper's internal world model for tracking object locations.

This module provides the HelperWorldModel, which tracks the helper agent's
own beliefs about object locations based PURELY on its own observations.
This eliminates data leakage where the helper previously accessed ground truth.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

from ..common.types import ObjectLocation, Observation


@dataclass
class HelperBelief:
    """Represents helper's belief about a single object."""
    location: ObjectLocation
    last_seen_timestamp: int
    uncertainty: float = 0.0  # 0.0 = certain, 1.0 = completely unknown


class HelperWorldModel:
    """Tracks helper's own beliefs about object locations.
    
    This model only updates based on what the HELPER sees. It does NOT
    monitor what the human sees (that's the job of the particle filter).
    This distinction is crucial for detecting false beliefs: 
    
    False Belief = Divergence between Human Belief (Particle Filter) 
                   and Reality (approximated by HelperWorldModel).
    """

    def __init__(self):
        self._beliefs: Dict[str, HelperBelief] = {}
        self._current_timestep = 0

    def update(self, observation: Observation, true_locations: Dict[str, ObjectLocation]) -> None:
        """Update beliefs based on helper's observation.
        
        Args:
            observation: The helper's current observation
            true_locations: True states from environment (used ONLY for visible objects)
        """
        self._current_timestep = observation.timestamp
        
        # We can only update beliefs for objects we can SEE
        visible_locs = {}
        if observation.visible_objects:
            for obj_id in observation.visible_objects:
                if obj_id in true_locations:
                    visible_locs[obj_id] = true_locations[obj_id]
        
        if visible_locs:
            self.update_from_perception(visible_locs)
            
    def update_from_perception(self, visible_objects: Dict[str, ObjectLocation]) -> None:
        """Update beliefs from detailed perception of visible objects.
        
        Args:
            visible_objects: Dictionary mapping object_id to ObjectLocation
                             for all objects currently visible to the helper.
        """
        for obj_id, loc in visible_objects.items():
            # When we see an object, we are certain about its location
            self._beliefs[obj_id] = HelperBelief(
                location=loc,
                last_seen_timestamp=self._current_timestep,
                uncertainty=0.1  # Small base uncertainty even when seen
            )

    def get_belief(self, object_id: str) -> Optional[HelperBelief]:
        """Get belief about a specific object."""
        return self._beliefs.get(object_id)

    def get_all_beliefs(self) -> Dict[str, ObjectLocation]:
        """Get best-guess locations for all known objects.
        
        Returns:
            Dictionary mapping object_id to ObjectLocation
        """
        return {
            obj_id: belief.location 
            for obj_id, belief in self._beliefs.items()
        }

    def get_confidence(self, object_id: str) -> float:
        """Get confidence in object location (1 - uncertainty)."""
        belief = self._beliefs.get(object_id)
        if not belief:
            return 0.0
            
        # Decay confidence over time?
        # For now, simple constant confidence if seen
        time_since_seen = self._current_timestep - belief.last_seen_timestamp
        
        # Simple decay: confidence drops by 0.01 per timestep
        decay = 0.01 * time_since_seen
        confidence = max(0.1, 1.0 - belief.uncertainty - decay)
        
        return confidence
