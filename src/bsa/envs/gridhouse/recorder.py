"""Episode recorder for serializing episodes to Parquet and JSONL."""

from pathlib import Path
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ...common.types import Episode, EpisodeStep, Action, ObjectLocation, Observation


class EpisodeRecorder:
    """Recorder for saving episodes to Parquet or JSONL format."""

    def save_episode(self, episode: Episode, output_path: Path, format: str = "parquet") -> None:
        """Save episode to file.

        Args:
            episode: Episode to save
            output_path: Output file path
            format: Format to use ('parquet' or 'jsonl')
        """
        if format == "parquet":
            self._save_parquet(episode, output_path)
        elif format == "jsonl":
            self._save_jsonl(episode, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _save_parquet(self, episode: Episode, output_path: Path) -> None:
        """Save episode to Parquet format."""
        # Convert episode steps to DataFrame rows
        rows = []
        for step in episode.steps:
            row = self._step_to_dict(step, episode)
            rows.append(row)

        df = pd.DataFrame(rows)

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to Parquet with compression
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_path, compression="snappy")

    def _save_jsonl(self, episode: Episode, output_path: Path) -> None:
        """Save episode to JSONL format."""
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def convert_to_json_serializable(obj):
            """Convert numpy types and other non-serializable objects to native Python types."""
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj

        with open(output_path, "w") as f:
            # Write episode metadata as first line
            metadata = convert_to_json_serializable({
                "episode_id": episode.episode_id,
                "goal_id": episode.goal_id,
                "tau": episode.tau,
                "intervention_type": episode.intervention_type,
                "metadata": episode.metadata,
                "type": "episode_metadata",
            })
            f.write(json.dumps(metadata) + "\n")

            # Write each step as a line
            for step in episode.steps:
                step_dict = self._step_to_dict(step, episode)
                step_dict = convert_to_json_serializable(step_dict)
                f.write(json.dumps(step_dict) + "\n")

    def load_episode(self, input_path: Path) -> Episode:
        """Load episode from Parquet file.

        Args:
            input_path: Path to Parquet file

        Returns:
            Loaded Episode object
        """
        df = pd.read_parquet(input_path)

        # Reconstruct episode from DataFrame
        episode_id = df["episode_id"].iloc[0]
        goal_id = df["goal_id"].iloc[0]
        tau = df["tau"].iloc[0] if pd.notna(df["tau"].iloc[0]) else None
        intervention_type = df["intervention_type"].iloc[0] if pd.notna(df["intervention_type"].iloc[0]) else None

        # Reconstruct steps
        steps = []
        for _, row in df.iterrows():
            step = self._dict_to_step(row)
            steps.append(step)

        # Reconstruct metadata from filename or defaults
        metadata = {
            "num_steps": len(steps),
            "tau": tau,
            "intervention_type": intervention_type,
        }

        # Check for false belief by comparing true vs believed locations
        false_belief_created = False
        false_belief_steps = 0
        for step in steps:
            for obj_id in step.true_object_locations:
                if obj_id in step.human_belief_object_locations:
                    true_loc = step.true_object_locations[obj_id]
                    belief_loc = step.human_belief_object_locations[obj_id]
                    if true_loc.room_id != belief_loc.room_id:
                        false_belief_created = True
                        false_belief_steps += 1
                        break

        metadata["false_belief_created"] = false_belief_created
        metadata["false_belief_steps"] = false_belief_steps

        # Extract condition from filename if present
        fname = input_path.stem
        for cond in ["control", "false_belief", "seen_relocation"]:
            if cond in fname:
                metadata["condition"] = cond
                break

        return Episode(
            episode_id=episode_id,
            goal_id=goal_id,
            tau=tau,
            intervention_type=intervention_type,
            steps=steps,
            metadata=metadata,
        )

    def _dict_to_step(self, row: Dict) -> EpisodeStep:
        """Convert dictionary row back to EpisodeStep."""
        # Reconstruct object locations from flattened columns
        true_locations = {}
        belief_locations = {}

        # Find all object IDs from column names
        obj_ids = set()
        for col in row.index:
            if col.startswith("true_") and col.endswith("_room"):
                obj_id = col[5:-5]  # Extract object_id from "true_{obj_id}_room"
                obj_ids.add(obj_id)

        for obj_id in obj_ids:
            # True locations
            true_room = row.get(f"true_{obj_id}_room")
            true_container = row.get(f"true_{obj_id}_container")
            true_x = row.get(f"true_{obj_id}_x", 0.0)
            true_y = row.get(f"true_{obj_id}_y", 0.0)
            true_z = row.get(f"true_{obj_id}_z", 0.0)

            if pd.notna(true_room):
                true_locations[obj_id] = ObjectLocation(
                    object_id=obj_id,
                    room_id=true_room,
                    container_id=true_container if pd.notna(true_container) else None,
                    position=(float(true_x), float(true_y), float(true_z)),
                )

            # Belief locations
            belief_room = row.get(f"belief_{obj_id}_room")
            belief_container = row.get(f"belief_{obj_id}_container")
            belief_x = row.get(f"belief_{obj_id}_x", 0.0)
            belief_y = row.get(f"belief_{obj_id}_y", 0.0)
            belief_z = row.get(f"belief_{obj_id}_z", 0.0)

            if pd.notna(belief_room):
                belief_locations[obj_id] = ObjectLocation(
                    object_id=obj_id,
                    room_id=belief_room,
                    container_id=belief_container if pd.notna(belief_container) else None,
                    position=(float(belief_x), float(belief_y), float(belief_z)),
                )

        # Helper to safely convert to list
        def to_list(val):
            if val is None:
                return []
            if isinstance(val, (list, tuple)):
                return list(val)
            if hasattr(val, 'tolist'):  # numpy array
                return val.tolist()
            return []

        # Reconstruct observations
        timestep = int(row.get("timestep", 0))
        human_obs = Observation(
            agent_id="human",
            current_room=row.get("human_obs_room", "") or "",
            position=(
                float(row.get("human_obs_x", 0) or 0),
                float(row.get("human_obs_y", 0) or 0),
                float(row.get("human_obs_z", 0) or 0),
            ),
            visible_objects=to_list(row.get("human_obs_visible_objects")),
            visible_containers=to_list(row.get("human_obs_visible_containers")),
            timestamp=timestep,
        )

        helper_obs = Observation(
            agent_id="helper",
            current_room=row.get("helper_obs_room", "") or "",
            position=(
                float(row.get("helper_obs_x", 0) or 0),
                float(row.get("helper_obs_y", 0) or 0),
                float(row.get("helper_obs_z", 0) or 0),
            ),
            visible_objects=to_list(row.get("helper_obs_visible_objects")),
            visible_containers=to_list(row.get("helper_obs_visible_containers")),
            timestamp=timestep,
        )

        # Parse action
        action_str = row.get("human_action", "WAIT")
        try:
            human_action = Action(action_str)
        except ValueError:
            human_action = Action.WAIT

        return EpisodeStep(
            episode_id=row.get("episode_id", ""),
            timestep=int(row.get("timestep", 0)),
            human_action=human_action,
            helper_obs=helper_obs,
            human_obs=human_obs,
            visible_objects_h=row.get("visible_objects_h", []),
            visible_objects_helper=row.get("visible_objects_helper", []),
            true_object_locations=true_locations,
            human_belief_object_locations=belief_locations,
            goal_id=row.get("goal_id", ""),
            tau=row.get("tau") if pd.notna(row.get("tau")) else None,
            intervention_type=row.get("intervention_type") if pd.notna(row.get("intervention_type")) else None,
        )

    def _step_to_dict(self, step: EpisodeStep, episode: Episode) -> Dict[str, Any]:
        """Convert EpisodeStep to dictionary for serialization."""
        # Flatten ObjectLocation structures
        true_locations_dict = {}
        belief_locations_dict = {}

        for obj_id, obj_loc in step.true_object_locations.items():
            true_locations_dict[f"true_{obj_id}_room"] = obj_loc.room_id
            true_locations_dict[f"true_{obj_id}_container"] = obj_loc.container_id
            true_locations_dict[f"true_{obj_id}_x"] = obj_loc.position[0]
            true_locations_dict[f"true_{obj_id}_y"] = obj_loc.position[1]
            true_locations_dict[f"true_{obj_id}_z"] = obj_loc.position[2]

        for obj_id, obj_loc in step.human_belief_object_locations.items():
            belief_locations_dict[f"belief_{obj_id}_room"] = obj_loc.room_id
            belief_locations_dict[f"belief_{obj_id}_container"] = obj_loc.container_id
            belief_locations_dict[f"belief_{obj_id}_x"] = obj_loc.position[0]
            belief_locations_dict[f"belief_{obj_id}_y"] = obj_loc.position[1]
            belief_locations_dict[f"belief_{obj_id}_z"] = obj_loc.position[2]

        # Flatten Observation structures
        human_obs_dict = {
            "human_obs_room": step.human_obs.current_room,
            "human_obs_x": step.human_obs.position[0],
            "human_obs_y": step.human_obs.position[1],
            "human_obs_z": step.human_obs.position[2],
            "human_obs_visible_objects": step.human_obs.visible_objects,
            "human_obs_visible_containers": step.human_obs.visible_containers,
        }

        helper_obs_dict = {
            "helper_obs_room": step.helper_obs.current_room,
            "helper_obs_x": step.helper_obs.position[0],
            "helper_obs_y": step.helper_obs.position[1],
            "helper_obs_z": step.helper_obs.position[2],
            "helper_obs_visible_objects": step.helper_obs.visible_objects,
            "helper_obs_visible_containers": step.helper_obs.visible_containers,
        }

        # Combine into single row
        row = {
            "episode_id": episode.episode_id,
            "timestep": step.timestep,
            "goal_id": step.goal_id,
            "tau": step.tau,
            "intervention_type": step.intervention_type,
            "human_action": step.human_action.value if isinstance(step.human_action, Action) else str(step.human_action),
            "visible_objects_h": step.visible_objects_h,
            "visible_objects_helper": step.visible_objects_helper,
            **human_obs_dict,
            **helper_obs_dict,
            **true_locations_dict,
            **belief_locations_dict,
        }

        return row
