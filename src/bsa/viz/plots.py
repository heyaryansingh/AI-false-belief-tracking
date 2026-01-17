"""Plotting module for generating visualizations from experiment results."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


class PlotGenerator:
    """Generator for creating publication-quality plots from experiment results."""

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Path,
    ):
        """Initialize plot generator.

        Args:
            df: Aggregated results DataFrame
            output_dir: Directory to save plots
        """
        self.df = df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_belief_timeline(
        self,
        episode_id: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot belief state timeline for an episode.

        Args:
            episode_id: Episode ID to plot (if None, plots first available)
            save_path: Path to save plot (if None, uses default)

        Returns:
            Path to saved plot
        """
        # For now, create a placeholder plot
        # In practice, would load episode data and plot belief evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Placeholder: plot aggregated goal inference accuracy over time
        if "goal_inference_accuracy_mean" in self.df.columns:
            models = self.df["group_model"].unique() if "group_model" in self.df.columns else []
            for model in models:
                model_df = self.df[self.df["group_model"] == model]
                if len(model_df) > 0:
                    ax.plot(
                        range(len(model_df)),
                        model_df["goal_inference_accuracy_mean"],
                        label=model,
                        marker='o',
                    )
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Goal Inference Accuracy")
        ax.set_title("Belief Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = self.output_dir / "belief_timeline.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path

    def plot_detection_auroc(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot AUROC comparison across models.

        Args:
            save_path: Path to save plot (if None, uses default)

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Filter for false_belief condition if available
        plot_df = self.df.copy()
        if "group_condition" in plot_df.columns:
            plot_df = plot_df[plot_df["group_condition"] == "false_belief"]
        
        if "group_model" in plot_df.columns and "false_belief_detection_auroc_mean" in plot_df.columns:
            models = plot_df["group_model"].unique()
            means = []
            stds = []
            labels = []
            
            for model in models:
                model_df = plot_df[plot_df["group_model"] == model]
                if len(model_df) > 0:
                    mean_val = model_df["false_belief_detection_auroc_mean"].iloc[0]
                    std_val = model_df["false_belief_detection_auroc_std"].iloc[0] if "false_belief_detection_auroc_std" in model_df.columns else 0.0
                    
                    if mean_val is not None and not np.isnan(mean_val):
                        means.append(mean_val)
                        stds.append(std_val if std_val is not None else 0.0)
                        labels.append(model)
            
            if means:
                x_pos = np.arange(len(labels))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels)
                ax.set_ylabel("AUROC")
                ax.set_title("False-Belief Detection AUROC by Model")
                ax.set_ylim([0, 1.1])
                ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, "No AUROC data available", ha='center', va='center')
            ax.set_title("False-Belief Detection AUROC")
        
        if save_path is None:
            save_path = self.output_dir / "detection_auroc.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path

    def plot_task_performance(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot task performance comparison across models/conditions.

        Args:
            save_path: Path to save plot (if None, uses default)

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Task completion rate
        ax1 = axes[0]
        if "group_model" in self.df.columns:
            models = self.df["group_model"].unique()
            completion_rates = []
            labels = []
            
            for model in models:
                model_df = self.df[self.df["group_model"] == model]
                if len(model_df) > 0 and "task_completed_mean" in model_df.columns:
                    rate = model_df["task_completed_mean"].iloc[0]
                    if rate is not None and not np.isnan(rate):
                        completion_rates.append(rate)
                        labels.append(model)
            
            if completion_rates:
                x_pos = np.arange(len(labels))
                ax1.bar(x_pos, completion_rates, alpha=0.7)
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(labels)
                ax1.set_ylabel("Completion Rate")
                ax1.set_title("Task Completion Rate by Model")
                ax1.set_ylim([0, 1.1])
                ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Wasted actions
        ax2 = axes[1]
        if "group_model" in self.df.columns:
            models = self.df["group_model"].unique()
            wasted_means = []
            wasted_stds = []
            labels = []
            
            for model in models:
                model_df = self.df[self.df["group_model"] == model]
                if len(model_df) > 0:
                    mean_val = model_df["num_wasted_actions_mean"].iloc[0] if "num_wasted_actions_mean" in model_df.columns else None
                    std_val = model_df["num_wasted_actions_std"].iloc[0] if "num_wasted_actions_std" in model_df.columns else None
                    
                    if mean_val is not None and not np.isnan(mean_val):
                        wasted_means.append(mean_val)
                        wasted_stds.append(std_val if std_val is not None else 0.0)
                        labels.append(model)
            
            if wasted_means:
                x_pos = np.arange(len(labels))
                ax2.bar(x_pos, wasted_means, yerr=wasted_stds, capsize=5, alpha=0.7)
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(labels)
                ax2.set_ylabel("Wasted Actions")
                ax2.set_title("Average Wasted Actions by Model")
                ax2.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = self.output_dir / "task_performance.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path

    def plot_ablation_curves(
        self,
        parameter: str,
        metric: str,
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Plot ablation curves showing parameter effect on metrics.

        Args:
            parameter: Parameter name to vary (e.g., 'num_particles')
            metric: Metric name to plot (e.g., 'false_belief_detection_auroc_mean')
            save_path: Path to save plot (if None, uses default)

        Returns:
            Path to saved plot or None if data not available
        """
        # Check if parameter column exists
        if parameter not in self.df.columns:
            return None
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Group by parameter value and plot metric
        if metric in self.df.columns:
            grouped = self.df.groupby(parameter)
            
            param_values = []
            metric_means = []
            metric_stds = []
            
            for param_val, group_df in grouped:
                if metric in group_df.columns:
                    mean_val = group_df[metric].mean()
                    std_val = group_df[metric].std()
                    
                    if mean_val is not None and not np.isnan(mean_val):
                        param_values.append(param_val)
                        metric_means.append(mean_val)
                        metric_stds.append(std_val if std_val is not None else 0.0)
            
            if param_values:
                ax.errorbar(param_values, metric_means, yerr=metric_stds, marker='o', capsize=5)
                ax.set_xlabel(parameter)
                ax.set_ylabel(metric)
                ax.set_title(f"Ablation: {parameter} vs {metric}")
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No ablation data available", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, f"Metric {metric} not available", ha='center', va='center')
        
        if save_path is None:
            save_path = self.output_dir / f"ablation_{parameter}_{metric}.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path

    def plot_intervention_quality(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot intervention quality metrics.

        Args:
            save_path: Path to save plot (if None, uses default)

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Precision/Recall
        ax1 = axes[0]
        if "group_model" in self.df.columns:
            models = self.df["group_model"].unique()
            precision_means = []
            recall_means = []
            labels = []
            
            for model in models:
                model_df = self.df[self.df["group_model"] == model]
                if len(model_df) > 0:
                    prec = model_df["intervention_precision_mean"].iloc[0] if "intervention_precision_mean" in model_df.columns else None
                    rec = model_df["intervention_recall_mean"].iloc[0] if "intervention_recall_mean" in model_df.columns else None
                    
                    if prec is not None and rec is not None and not (np.isnan(prec) or np.isnan(rec)):
                        precision_means.append(prec)
                        recall_means.append(rec)
                        labels.append(model)
            
            if precision_means:
                x_pos = np.arange(len(labels))
                width = 0.35
                ax1.bar(x_pos - width/2, precision_means, width, label='Precision', alpha=0.7)
                ax1.bar(x_pos + width/2, recall_means, width, label='Recall', alpha=0.7)
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(labels)
                ax1.set_ylabel("Score")
                ax1.set_title("Intervention Precision/Recall by Model")
                ax1.set_ylim([0, 1.1])
                ax1.legend()
                ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Over/Under-correction
        ax2 = axes[1]
        if "group_model" in self.df.columns:
            models = self.df["group_model"].unique()
            over_means = []
            under_means = []
            labels = []
            
            for model in models:
                model_df = self.df[self.df["group_model"] == model]
                if len(model_df) > 0:
                    over = model_df["over_corrections_mean"].iloc[0] if "over_corrections_mean" in model_df.columns else None
                    under = model_df["under_corrections_mean"].iloc[0] if "under_corrections_mean" in model_df.columns else None
                    
                    if over is not None and under is not None and not (np.isnan(over) or np.isnan(under)):
                        over_means.append(over)
                        under_means.append(under)
                        labels.append(model)
            
            if over_means:
                x_pos = np.arange(len(labels))
                width = 0.35
                ax2.bar(x_pos - width/2, over_means, width, label='Over-correction', alpha=0.7)
                ax2.bar(x_pos + width/2, under_means, width, label='Under-correction', alpha=0.7)
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(labels)
                ax2.set_ylabel("Count")
                ax2.set_title("Over/Under-Correction by Model")
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
        
        if save_path is None:
            save_path = self.output_dir / "intervention_quality.png"
        else:
            save_path = Path(save_path)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path


def generate_plots(
    config: Dict[str, Any],
    aggregated_df: Optional[pd.DataFrame] = None,
) -> List[Path]:
    """Generate plots from config (called by analysis pipeline).

    Args:
        config: Analysis configuration dictionary
        aggregated_df: Aggregated DataFrame (if None, loads from config)

    Returns:
        List of paths to generated plots
    """
    # Get output directory from config
    analysis_config = config.get("analysis", {})
    output_dir = Path(analysis_config.get("output_dir", "results/figures"))
    
    # Load aggregated results if not provided
    if aggregated_df is None:
        from ..analysis.aggregate import AnalysisAggregator
        aggregator = AnalysisAggregator()
        input_dir = Path(analysis_config.get("input_dir", "results/metrics"))
        df = aggregator.load_results(input_dir=input_dir)
        aggregated_df = aggregator.aggregate_metrics(df)
    
    # Create plot generator
    plotter = PlotGenerator(aggregated_df, output_dir)
    
    # Get plot specifications from config
    plots_config = config.get("plots", [])
    
    generated_plots = []
    
    # Generate each plot
    for plot_spec in plots_config:
        plot_type = plot_spec.get("type")
        filename = plot_spec.get("filename", f"{plot_type}.png")
        save_path = output_dir / filename
        
        print(f"  Generating {plot_type} plot...")
        
        if plot_type == "belief_timeline":
            plot_path = plotter.plot_belief_timeline(save_path=save_path)
        elif plot_type == "detection_auroc":
            plot_path = plotter.plot_detection_auroc(save_path=save_path)
        elif plot_type == "task_performance":
            plot_path = plotter.plot_task_performance(save_path=save_path)
        elif plot_type == "intervention_quality":
            plot_path = plotter.plot_intervention_quality(save_path=save_path)
        elif plot_type == "ablation_curves":
            parameter = plot_spec.get("parameter", "num_particles")
            metric = plot_spec.get("metric", "false_belief_detection_auroc_mean")
            plot_path = plotter.plot_ablation_curves(parameter, metric, save_path=save_path)
        else:
            print(f"    [WARN] Unknown plot type: {plot_type}")
            continue
        
        if plot_path:
            generated_plots.append(plot_path)
            print(f"    [OK] Saved to: {plot_path}")
    
    return generated_plots
