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
