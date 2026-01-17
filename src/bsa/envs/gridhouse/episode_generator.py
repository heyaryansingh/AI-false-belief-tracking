"""Episode generator for GridHouse."""

from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from ...common.types import Episode, EpisodeStep, Goal, Action, ObjectLocation
from ...common.seeding import get_rng
from ...agents.human import ScriptedHumanAgent
from .env import GridHouseEnvironment
from .tasks import get_task, list_tasks
from .recorder import EpisodeRecorder


class GridHouseEpisodeGenerator:
    """Generate episodes with false-belief interventions."""

    def __init__(
        self,
        env: GridHouseEnvironment,
        seed: Optional[int] = None,
        tau_range: tuple[int, int] = (5, 20),
        occlusion_severity: float = 0.5,
        drift_probability: float = 0.5,
    ):
        """Initialize episode generator.

        Args:
            env: GridHouse environment instance
            seed: Random seed
            tau_range: Range of intervention timesteps
            occlusion_severity: Severity of occlusion (0-1)
            drift_probability: Probability of false-belief intervention
        """
        self.env = env
        self.rng = get_rng(seed)
        self.tau_range = tau_range
        self.occlusion_severity = occlusion_severity
        self.drift_probability = drift_probability

    def generate_episode(
        self,
        goal_id: Optional[str] = None,
        tau: Optional[int] = None,
        intervention_type: Optional[str] = None,
        save_path: Optional[Path] = None,
        format: str = "parquet",
    ) -> Episode:
        """Generate a single episode.

        Args:
            goal_id: Task goal ID (if None, randomly selected)
            tau: Intervention timestep (if None, randomly selected)
            intervention_type: Type of intervention (if None, randomly selected)

        Returns:
            Generated episode
        """
        # Reset environment
        self.env.reset(seed=self.rng.integers(0, 2**31))

        # Select goal
        if goal_id is None:
            goal_id = self.rng.choice(list_tasks())
        task = get_task(goal_id)

        # Validate task has critical objects
        if not task.critical_objects:
            raise ValueError(f"Task {goal_id} has no critical objects")

        # Select intervention parameters
        if tau is None:
            tau = self.rng.integers(*self.tau_range)
        if intervention_type is None:
            intervention_type = "relocate" if self.rng.random() < self.drift_probability else None

        # Initialize human agent with goal
        initial_beliefs = self.env.get_object_locations().copy()
        human_agent = ScriptedHumanAgent(goal=task, initial_belief_state=initial_beliefs)

        # Generate episode steps
        steps: List[EpisodeStep] = []
        max_steps = 50

        # Track human belief state
        human_belief_locations = initial_beliefs.copy()

        for t in range(max_steps):
            # Get human observation
            human_obs = self.env.get_visible_state("human")

            # Apply intervention at tau
            if t == tau and intervention_type == "relocate":
                self._apply_intervention(task, human_belief_locations)

            # Human action using agent's planning
            human_action = human_agent.plan_action(
                observation=human_obs,
                task=task,
                belief_state=human_belief_locations,
            )

            # Execute action
            human_obs_after, _, done, _ = self.env.step(human_action, "human")

            # Update human belief (only if they observe)
            true_locations = self.env.get_object_locations()
            human_agent.update_belief(
                observation=human_obs_after,
                true_object_locations=true_locations,
                belief_state=human_belief_locations,
            )

            # Helper observation
            helper_obs = self.env.get_visible_state("helper")

            # Create step
            step = EpisodeStep(
                episode_id=f"ep_{len(steps)}",
                timestep=t,
                human_action=human_action,
                helper_obs=helper_obs,
                human_obs=human_obs_after,
                visible_objects_h=human_obs_after.visible_objects,
                visible_objects_helper=helper_obs.visible_objects,
                true_object_locations=true_locations,
                human_belief_object_locations=human_belief_locations.copy(),
                goal_id=goal_id,
                tau=tau if t >= tau else None,
                intervention_type=intervention_type if t >= tau else None,
            )
            steps.append(step)

            if done:
                break

        # Validate episode
        if len(steps) < tau + 5:
            # Episode too short - log warning but continue
            pass

        # Validate intervention occurred if expected
        intervention_applied = False
        false_belief_created = False
        if intervention_type == "relocate" and tau is not None:
            # Check if intervention was applied (object moved)
            if len(steps) > tau:
                step_after = steps[tau]
                # Check if any critical object moved
                for obj_id in task.critical_objects:
                    if obj_id in step_after.true_object_locations:
                        true_loc = step_after.true_object_locations[obj_id]
                        belief_loc = step_after.human_belief_object_locations.get(obj_id)
                        if belief_loc and true_loc.room_id != belief_loc.room_id:
                            intervention_applied = True
                            false_belief_created = True
                            break

        # Update metadata with validation results
        metadata = {
            "max_steps": max_steps,
            "occlusion_severity": self.occlusion_severity,
            "intervention_applied": intervention_applied,
            "false_belief_created": false_belief_created,
            "task_completed": False,  # TODO: Implement task completion detection
            "num_steps": len(steps),
        }

        episode = Episode(
            episode_id=f"episode_{self.rng.integers(0, 1000000)}",
            goal_id=goal_id,
            tau=tau,
            intervention_type=intervention_type,
            steps=steps,
            metadata=metadata,
        )

        # Save if path provided
        if save_path:
            recorder = EpisodeRecorder()
            recorder.save_episode(episode, save_path, format=format)

        return episode

    def generate_episodes(
        self,
        num_episodes: int,
        output_dir: Path,
        format: str = "parquet",
        goal_distribution: Optional[List[str]] = None,
    ) -> List[Episode]:
        """Generate multiple episodes and save them.

        Args:
            num_episodes: Number of episodes to generate
            output_dir: Directory to save episodes
            format: Format to use ('parquet' or 'jsonl')
            goal_distribution: List of goal IDs to sample from (if None, uses all tasks)

        Returns:
            List of generated episodes
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        recorder = EpisodeRecorder()
        episodes = []

        if goal_distribution is None:
            goal_distribution = list_tasks()

        ext = "parquet" if format == "parquet" else "jsonl"

        for i in range(num_episodes):
            # Sample goal from distribution
            goal_id = self.rng.choice(goal_distribution)

            # Generate episode
            episode = self.generate_episode(goal_id=goal_id)

            # Save episode
            output_path = output_dir / f"episode_{episode.episode_id}.{ext}"
            recorder.save_episode(episode, output_path, format=format)

            episodes.append(episode)

            # Progress logging
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_episodes} episodes")

        return episodes

    def _apply_intervention(self, task, human_belief_locations: Dict) -> None:
        """Apply false-belief intervention.

        Moves a critical object to a different location while the human agent
        cannot see, creating a false-belief scenario.

        Args:
            task: Task being performed
            human_belief_locations: Human's belief state (NOT updated by this method)
        """
        # Get human's current position and room
        human_pos = self.env.agent_positions.get("human")
        human_room = self.env.agent_rooms.get("human")

        if human_pos is None or human_room is None:
            return  # Human not initialized

        # Find a critical object to relocate
        for obj_id in task.critical_objects:
            if obj_id not in self.env.object_locations:
                continue  # Object doesn't exist

            obj_loc = self.env.object_locations[obj_id]

            # Check if human can see this object
            # Human can see if: same room AND within visibility radius
            can_see = False
            if obj_loc.room_id == human_room:
                distance = np.sqrt(
                    (obj_loc.position[0] - human_pos[0]) ** 2 + (obj_loc.position[1] - human_pos[1]) ** 2
                )
                if distance <= self.env.visibility_radius:
                    # Also check if object is in container - only visible if container is open
                    if obj_loc.container_id:
                        container = self.env.containers.get(obj_loc.container_id)
                        if container and container.is_open:
                            can_see = True
                    else:
                        can_see = True

            # Only relocate if human cannot see
            if not can_see:
                # Find alternative location (different room or different container)
                old_container_id = obj_loc.container_id
                old_room_id = obj_loc.room_id

                # Try to find alternative container in different room
                alternative_container = None
                for container_id, container in self.env.containers.items():
                    if container.room_id != old_room_id:
                        alternative_container = container
                        break

                if alternative_container:
                    # Move to alternative container
                    new_container_id = alternative_container.container_id
                    new_room_id = alternative_container.room_id
                    new_position = (
                        float(alternative_container.position[0]),
                        float(alternative_container.position[1]),
                        0.0,
                    )

                    # Remove from old container
                    if old_container_id:
                        old_container = self.env.containers.get(old_container_id)
                        if old_container and obj_id in old_container.contents:
                            old_container.contents.remove(obj_id)

                    # Add to new container
                    alternative_container.contents.append(obj_id)

                    # Update object location
                    self.env.object_locations[obj_id] = ObjectLocation(
                        object_id=obj_id,
                        container_id=new_container_id,
                        room_id=new_room_id,
                        position=new_position,
                    )
                else:
                    # No alternative container - move to different room
                    alternative_rooms = [rid for rid in self.env.rooms.keys() if rid != old_room_id]
                    if alternative_rooms:
                        new_room_id = self.rng.choice(alternative_rooms)
                        new_room = self.env.rooms[new_room_id]

                        # Place in center of new room
                        x = (new_room.bounds[0] + new_room.bounds[2]) // 2
                        y = (new_room.bounds[1] + new_room.bounds[3]) // 2
                        new_position = (float(x), float(y), 0.0)

                        # Remove from old container if needed
                        if old_container_id:
                            old_container = self.env.containers.get(old_container_id)
                            if old_container and obj_id in old_container.contents:
                                old_container.contents.remove(obj_id)

                        # Update object location
                        self.env.object_locations[obj_id] = ObjectLocation(
                            object_id=obj_id,
                            container_id=None,
                            room_id=new_room_id,
                            position=new_position,
                        )

                # Important: Do NOT update human_belief_locations
                # This creates the false belief - human still believes object at old location
                break  # Only relocate one object per intervention

