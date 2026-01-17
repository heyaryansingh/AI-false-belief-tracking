"""Observability module for VirtualHome environments.

Provides enhanced observability functions for debugging and analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from ...common.types import Episode, EpisodeStep, Observation, ObjectLocation
from .env import VirtualHomeEnvironment


def get_scene_state(env: VirtualHomeEnvironment) -> Dict[str, Any]:
    """Get full scene state from VirtualHome environment.
    
    Args:
        env: VirtualHome environment instance
        
    Returns:
        Dictionary containing scene state (rooms, objects, agents, containers)
    """
    true_state = env.get_true_state()
    
    # Extract structured scene information
    scene_state = {
        "rooms": true_state.get("available_rooms", []),
        "objects": env.get_object_locations(),
        "agents": true_state.get("agent_positions", {}),
        "agent_rooms": true_state.get("agent_rooms", {}),
        "timestep": true_state.get("timestep", 0),
    }
    
    return scene_state


def get_agent_view(env: VirtualHomeEnvironment, agent_id: str) -> Dict[str, Any]:
    """Get what an agent can see (objects, containers, rooms).
    
    Args:
        env: VirtualHome environment instance
        agent_id: ID of agent
        
    Returns:
        Dictionary containing agent's view (observation + context)
    """
    obs = env.get_visible_state(agent_id)
    true_state = env.get_true_state()
    
    agent_view = {
        "agent_id": agent_id,
        "current_room": obs.current_room,
        "position": obs.position,
        "visible_objects": obs.visible_objects,
        "visible_containers": obs.visible_containers,
        "timestamp": obs.timestamp,
        # Additional context
        "all_objects_in_scene": list(env.get_object_locations().keys()),
        "all_rooms": true_state.get("available_rooms", []),
    }
    
    return agent_view


def get_object_trajectory(object_id: str, episode: Episode) -> List[Dict[str, Any]]:
    """Track object movement over episode.
    
    Args:
        object_id: ID of object to track
        episode: Episode to analyze
        
    Returns:
        List of object locations at each timestep
    """
    trajectory = []
    
    for step in episode.steps:
        true_loc = step.true_object_locations.get(object_id)
        belief_loc = step.human_belief_object_locations.get(object_id)
        
        if true_loc:
            trajectory.append({
                "timestep": step.timestep,
                "true_location": {
                    "room_id": true_loc.room_id,
                    "container_id": true_loc.container_id,
                    "position": true_loc.position,
                },
                "human_belief_location": {
                    "room_id": belief_loc.room_id if belief_loc else None,
                    "container_id": belief_loc.container_id if belief_loc else None,
                    "position": belief_loc.position if belief_loc else None,
                } if belief_loc else None,
                "visible_to_human": object_id in step.visible_objects_h,
                "visible_to_helper": object_id in step.visible_objects_helper,
            })
    
    return trajectory


def analyze_observability(episode: Episode) -> Dict[str, Any]:
    """Analyze what each agent observed vs. true state.
    
    Args:
        episode: Episode to analyze
        
    Returns:
        Dictionary containing observability analysis
    """
    analysis = {
        "episode_id": episode.episode_id,
        "goal_id": episode.goal_id,
        "num_steps": len(episode.steps),
        "human_observations": [],
        "helper_observations": [],
        "false_beliefs": [],
    }
    
    # Track all objects in episode
    all_objects = set()
    for step in episode.steps:
        all_objects.update(step.true_object_locations.keys())
    
    # Analyze each step
    for step in episode.steps:
        # Human observability
        human_obs = {
            "timestep": step.timestep,
            "visible_objects": step.visible_objects_h,
            "visible_containers": step.human_obs.visible_containers,
            "current_room": step.human_obs.current_room,
        }
        analysis["human_observations"].append(human_obs)
        
        # Helper observability
        helper_obs = {
            "timestep": step.timestep,
            "visible_objects": step.visible_objects_helper,
            "visible_containers": step.helper_obs.visible_containers,
            "current_room": step.helper_obs.current_room,
        }
        analysis["helper_observations"].append(helper_obs)
        
        # Detect false beliefs
        for obj_id in all_objects:
            true_loc = step.true_object_locations.get(obj_id)
            belief_loc = step.human_belief_object_locations.get(obj_id)
            
            if true_loc and belief_loc:
                # Check if belief differs from truth
                if true_loc.room_id != belief_loc.room_id:
                    analysis["false_beliefs"].append({
                        "timestep": step.timestep,
                        "object_id": obj_id,
                        "true_room": true_loc.room_id,
                        "believed_room": belief_loc.room_id,
                        "visible_to_human": obj_id in step.visible_objects_h,
                    })
    
    # Summary statistics
    analysis["summary"] = {
        "total_objects": len(all_objects),
        "false_belief_count": len(analysis["false_beliefs"]),
        "intervention_timestep": episode.tau,
        "intervention_type": episode.intervention_type,
    }
    
    return analysis


def visualize_episode(episode: Episode, output_path: Optional[str] = None) -> str:
    """Generate text-based visualization of episode.
    
    Args:
        episode: Episode to visualize
        output_path: Optional path to save visualization (if None, returns string)
        
    Returns:
        Visualization string
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"Episode Visualization: {episode.episode_id}")
    lines.append("=" * 70)
    lines.append(f"Goal: {episode.goal_id}")
    lines.append(f"Intervention: {episode.intervention_type} at timestep {episode.tau}")
    lines.append(f"Total Steps: {len(episode.steps)}")
    lines.append("")
    
    # Show key steps
    for i, step in enumerate(episode.steps):
        if i == 0 or i == episode.tau or i == len(episode.steps) - 1:
            lines.append(f"Step {step.timestep}:")
            lines.append(f"  Human Action: {step.human_action.value}")
            lines.append(f"  Human Room: {step.human_obs.current_room}")
            lines.append(f"  Visible Objects: {step.visible_objects_h}")
            
            if step.tau is not None:
                lines.append(f"  [INTERVENTION AT THIS STEP]")
            lines.append("")
    
    visualization = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(visualization)
    
    return visualization
