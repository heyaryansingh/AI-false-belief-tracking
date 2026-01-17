# Data Directory

This directory contains generated episodes and datasets.

## Structure

- `episodes/` - Generated episode files (Parquet/JSONL format)

## Episode Schema

Each episode file contains:
- `episode_id`: Unique identifier
- `t`: Timestep
- `human_action`: Action taken by human agent
- `helper_obs`: Observation available to helper
- `human_obs`: Observation available to human
- `visible_objects_h`: Objects visible to human
- `visible_objects_helper`: Objects visible to helper
- `true_object_locations`: True object locations
- `human_belief_object_locations`: Human's belief about object locations
- `goal_id`: Task goal identifier
- `tau`: Intervention timestep (if applicable)
- `intervention_type`: Type of intervention (if applicable)
