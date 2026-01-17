"""Aggregate and analyze results."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import pandas as pd
import numpy as np
from scipy import stats
import pyarrow as pa
import pyarrow.parquet as pq

from ..common.config import load_config


class AnalysisAggregator:
    """Aggregator for loading and aggregating experiment results."""

    def __init__(self):
        """Initialize analysis aggregator."""
        pass

    def load_results(
        self,
        input_path: Optional[Path] = None,
        input_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """Load results from Parquet file(s).

        Args:
            input_path: Path to single Parquet file
            input_dir: Directory containing Parquet files (loads all)

        Returns:
            DataFrame with all results
        """
        if input_path:
            if not input_path.exists():
                raise FileNotFoundError(f"Results file not found: {input_path}")
            return pd.read_parquet(input_path)
        
        if input_dir:
            input_dir = Path(input_dir)
            if not input_dir.exists():
                raise FileNotFoundError(f"Results directory not found: {input_dir}")
            
            # Find all Parquet files in directory and subdirectories
            parquet_files = list(input_dir.rglob("results.parquet"))
            
            if not parquet_files:
                raise ValueError(f"No Parquet files found in {input_dir}")
            
            # Load and concatenate all results
            dfs = []
            for file_path in parquet_files:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            
            return pd.concat(dfs, ignore_index=True)
        
        # Default: load from results/metrics/
        default_dir = Path("results/metrics")
        return self.load_results(input_dir=default_dir)

    def aggregate_metrics(
        self,
        df: pd.DataFrame,
        group_by: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate metrics across runs/models/conditions.

        Args:
            df: Results DataFrame
            group_by: Columns to group by (default: ['model', 'condition'])

        Returns:
            Aggregated DataFrame with statistics
        """
        if group_by is None:
            group_by = ["model", "condition"]
        
        # Ensure group_by columns exist
        group_by = [col for col in group_by if col in df.columns]
        
        if not group_by:
            # No grouping - aggregate all rows
            return self._compute_statistics(df)
        
        # Group and aggregate
        grouped = df.groupby(group_by)
        
        # Get numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Compute statistics for each group
        agg_results = []
        for name, group_df in grouped:
            stats_dict = self._compute_statistics(group_df)
            
            # Add group identifiers
            if isinstance(name, tuple):
                for i, col in enumerate(group_by):
                    stats_dict[f"group_{col}"] = name[i]
            else:
                stats_dict[f"group_{group_by[0]}"] = name
            
            agg_results.append(stats_dict)
        
        return pd.DataFrame(agg_results)

    def _compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics for a DataFrame.

        Args:
            df: DataFrame to compute statistics for

        Returns:
            Dictionary with statistics for each numeric column
        """
        stats_dict = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            values = df[col].dropna()
            
            if len(values) == 0:
                stats_dict[f"{col}_mean"] = None
                stats_dict[f"{col}_std"] = None
                stats_dict[f"{col}_min"] = None
                stats_dict[f"{col}_max"] = None
                stats_dict[f"{col}_ci_lower"] = None
                stats_dict[f"{col}_ci_upper"] = None
                stats_dict[f"{col}_count"] = 0
                continue
            
            mean = float(values.mean())
            std = float(values.std()) if len(values) > 1 else 0.0
            min_val = float(values.min())
            max_val = float(values.max())
            count = len(values)
            
            # Compute 95% confidence interval
            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
                ci_lower = float(ci[0])
                ci_upper = float(ci[1])
            else:
                ci_lower = mean
                ci_upper = mean
            
            stats_dict[f"{col}_mean"] = mean
            stats_dict[f"{col}_std"] = std
            stats_dict[f"{col}_min"] = min_val
            stats_dict[f"{col}_max"] = max_val
            stats_dict[f"{col}_ci_lower"] = ci_lower
            stats_dict[f"{col}_ci_upper"] = ci_upper
            stats_dict[f"{col}_count"] = count
        
        return stats_dict

    def compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute overall summary statistics.

        Args:
            df: Results DataFrame

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "total_runs": len(df),
            "models": df["model"].unique().tolist() if "model" in df.columns else [],
            "conditions": df["condition"].unique().tolist() if "condition" in df.columns else [],
            "model_comparison": {},
            "condition_comparison": {},
        }
        
        # Model comparison
        if "model" in df.columns:
            for model in df["model"].unique():
                model_df = df[df["model"] == model]
                summary["model_comparison"][model] = {
                    "num_runs": len(model_df),
                    "task_completion_rate": self._compute_completion_rate(model_df),
                    "avg_detection_auroc": self._safe_mean(model_df, "false_belief_detection_auroc"),
                    "avg_detection_latency": self._safe_mean(model_df, "false_belief_detection_latency"),
                }
        
        # Condition comparison
        if "condition" in df.columns:
            for condition in df["condition"].unique():
                condition_df = df[df["condition"] == condition]
                summary["condition_comparison"][condition] = {
                    "num_runs": len(condition_df),
                    "task_completion_rate": self._compute_completion_rate(condition_df),
                    "avg_wasted_actions": self._safe_mean(condition_df, "num_wasted_actions"),
                }
        
        return summary

    def filter_results(
        self,
        df: pd.DataFrame,
        model: Optional[str] = None,
        condition: Optional[str] = None,
        goal_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter results by criteria.

        Args:
            df: Results DataFrame
            model: Filter by model name
            condition: Filter by condition
            goal_id: Filter by goal ID

        Returns:
            Filtered DataFrame
        """
        filtered = df.copy()
        
        if model is not None and "model" in filtered.columns:
            filtered = filtered[filtered["model"] == model]
        
        if condition is not None and "condition" in filtered.columns:
            filtered = filtered[filtered["condition"] == condition]
        
        if goal_id is not None and "goal_id" in filtered.columns:
            filtered = filtered[filtered["goal_id"] == goal_id]
        
        return filtered

    def _compute_completion_rate(self, df: pd.DataFrame) -> float:
        """Compute task completion rate.

        Args:
            df: Results DataFrame

        Returns:
            Completion rate (0-1)
        """
        if "task_completed" not in df.columns:
            return 0.0
        
        completed = df["task_completed"].sum()
        total = len(df)
        return float(completed / total) if total > 0 else 0.0

    def _safe_mean(self, df: pd.DataFrame, col: str) -> Optional[float]:
        """Safely compute mean, handling NaN values.

        Args:
            df: DataFrame
            col: Column name

        Returns:
            Mean value or None if column doesn't exist or all NaN
        """
        if col not in df.columns:
            return None
        
        values = df[col].dropna()
        if len(values) == 0:
            return None
        
        return float(values.mean())

    def _aggregate_detection_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate false-belief detection metrics.

        Args:
            df: Results DataFrame

        Returns:
            Dictionary with aggregated detection metrics
        """
        return {
            "auroc_mean": self._safe_mean(df, "false_belief_detection_auroc"),
            "auroc_std": self._safe_std(df, "false_belief_detection_auroc"),
            "latency_mean": self._safe_mean(df, "false_belief_detection_latency"),
            "latency_std": self._safe_std(df, "false_belief_detection_latency"),
            "fpr_mean": self._safe_mean(df, "false_belief_detection_fpr"),
            "fpr_std": self._safe_std(df, "false_belief_detection_fpr"),
        }

    def _aggregate_belief_tracking_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate belief tracking metrics.

        Args:
            df: Results DataFrame

        Returns:
            Dictionary with aggregated belief tracking metrics
        """
        return {
            "goal_accuracy_mean": self._safe_mean(df, "goal_inference_accuracy"),
            "goal_accuracy_std": self._safe_std(df, "goal_inference_accuracy"),
            "cross_entropy_mean": self._safe_mean(df, "belief_tracking_cross_entropy"),
            "cross_entropy_std": self._safe_std(df, "belief_tracking_cross_entropy"),
            "brier_score_mean": self._safe_mean(df, "belief_tracking_brier_score"),
            "brier_score_std": self._safe_std(df, "belief_tracking_brier_score"),
        }

    def _aggregate_task_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate task performance metrics.

        Args:
            df: Results DataFrame

        Returns:
            Dictionary with aggregated task performance metrics
        """
        return {
            "completion_rate": self._compute_completion_rate(df),
            "steps_mean": self._safe_mean(df, "num_steps_to_completion"),
            "steps_std": self._safe_std(df, "num_steps_to_completion"),
            "wasted_actions_mean": self._safe_mean(df, "num_wasted_actions"),
            "wasted_actions_std": self._safe_std(df, "num_wasted_actions"),
            "efficiency_mean": self._safe_mean(df, "task_efficiency"),
            "efficiency_std": self._safe_std(df, "task_efficiency"),
        }

    def _aggregate_intervention_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate intervention quality metrics.

        Args:
            df: Results DataFrame

        Returns:
            Dictionary with aggregated intervention metrics
        """
        return {
            "interventions_mean": self._safe_mean(df, "num_interventions"),
            "interventions_std": self._safe_std(df, "num_interventions"),
            "over_corrections_mean": self._safe_mean(df, "over_corrections"),
            "over_corrections_std": self._safe_std(df, "over_corrections"),
            "under_corrections_mean": self._safe_mean(df, "under_corrections"),
            "under_corrections_std": self._safe_std(df, "under_corrections"),
            "precision_mean": self._safe_mean(df, "intervention_precision"),
            "precision_std": self._safe_std(df, "intervention_precision"),
            "recall_mean": self._safe_mean(df, "intervention_recall"),
            "recall_std": self._safe_std(df, "intervention_recall"),
        }

    def _safe_std(self, df: pd.DataFrame, col: str) -> Optional[float]:
        """Safely compute std, handling NaN values.

        Args:
            df: DataFrame
            col: Column name

        Returns:
            Std value or None if column doesn't exist or all NaN
        """
        if col not in df.columns:
            return None
        
        values = df[col].dropna()
        if len(values) <= 1:
            return None
        
        return float(values.std())


def aggregate_results(
    config: Dict[str, Any],
    input_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate experiment results (called by CLI).

    Args:
        config: Analysis configuration dictionary
        input_dir: Input directory for results (overrides config)

    Returns:
        Tuple of (aggregated DataFrame, summary statistics dictionary)
    """
    # Get input directory from config or use default
    if input_dir is None:
        analysis_config = config.get("analysis", {})
        input_dir = Path(analysis_config.get("input_dir", "results/metrics"))
    
    # Get output directory
    analysis_config = config.get("analysis", {})
    output_dir = Path(analysis_config.get("output_dir", "results/metrics/aggregated"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {input_dir}")
    
    # Create aggregator and load results
    aggregator = AnalysisAggregator()
    df = aggregator.load_results(input_dir=input_dir)
    
    print(f"Loaded {len(df)} result rows")
    print(f"  Models: {df['model'].unique().tolist() if 'model' in df.columns else 'N/A'}")
    print(f"  Conditions: {df['condition'].unique().tolist() if 'condition' in df.columns else 'N/A'}")
    
    # Aggregate metrics
    print("\nAggregating metrics...")
    agg_df = aggregator.aggregate_metrics(df)
    
    # Compute summary statistics
    print("Computing summary statistics...")
    summary = aggregator.compute_summary_statistics(df)
    
    # Save aggregated results
    print(f"\nSaving aggregated results to: {output_dir}")
    
    # Save as Parquet
    parquet_path = output_dir / "aggregated_results.parquet"
    table = pa.Table.from_pandas(agg_df)
    pq.write_table(table, parquet_path, compression="snappy")
    print(f"  Parquet: {parquet_path}")
    
    # Save summary as JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {summary_path}")
    
    return agg_df, summary


def analyze_results(config: Dict[str, Any], input_dir: Optional[Path] = None) -> None:
    """Analyze experiment results (main entry point for CLI).

    Args:
        config: Analysis configuration dictionary
        input_dir: Input directory for results (overrides config)
    """
    print("=" * 70)
    print("Analysis Pipeline".center(70))
    print("=" * 70)
    print()
    
    # Step 1: Aggregate results
    print("[1/4] Aggregating results...")
    try:
        agg_df, summary = aggregate_results(config, input_dir)
        print("  [OK] Aggregation complete")
    except Exception as e:
        print(f"  [FAIL] Aggregation failed: {e}")
        return
    
    # Steps 2-4 will be implemented in later plans
    print("\n[2/4] Generating plots...")
    print("  [INFO] Plot generation will be implemented in Plan 05-02")
    
    print("\n[3/4] Generating tables...")
    print("  [INFO] Table generation will be implemented in Plan 05-03")
    
    print("\n[4/4] Generating report...")
    print("  [INFO] Report generation will be implemented in Plan 05-04")
    
    print("\n" + "=" * 70)
    print("Analysis complete (aggregation only - plots/tables/report pending)")
    print("=" * 70)
