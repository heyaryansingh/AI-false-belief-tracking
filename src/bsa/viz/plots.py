"""Plotting module for generating visualizations from experiment results."""

from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
plt.rcParams['font.family'] = 'sans-serif'

# Professional color palettes
MODEL_COLORS = {
    'reactive': '#E74C3C',      # Red
    'goal_only': '#3498DB',     # Blue
    'belief_pf': '#27AE60',     # Green
}

CONDITION_COLORS = {
    'control': '#95A5A6',       # Gray
    'false_belief': '#E74C3C', # Red
    'seen_relocation': '#9B59B6',  # Purple
}


class PlotGenerator:
    """Generator for creating publication-quality plots from experiment results."""

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        raw_df: Optional[pd.DataFrame] = None,
    ):
        """Initialize plot generator.

        Args:
            df: Aggregated results DataFrame
            output_dir: Directory to save plots
            raw_df: Raw (non-aggregated) results DataFrame for detailed plots
        """
        self.df = df
        self.raw_df = raw_df
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_color(self, model: str) -> str:
        """Get color for a model."""
        return MODEL_COLORS.get(model, '#7F8C8D')

    def _get_condition_color(self, condition: str) -> str:
        """Get color for a condition."""
        return CONDITION_COLORS.get(condition, '#7F8C8D')

    def _add_significance_stars(
        self,
        ax: plt.Axes,
        x1: float,
        x2: float,
        y: float,
        p_value: float,
        height: float = 0.02,
    ) -> None:
        """Add significance stars between two bars."""
        if p_value >= 0.05:
            return

        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        else:
            stars = "*"

        # Draw bracket
        ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], 'k-', lw=1)
        ax.text((x1 + x2) / 2, y + height, stars, ha='center', va='bottom', fontsize=10)

    # =========================================================================
    # DETAILED AUROC PLOTS
    # =========================================================================

    def plot_detection_auroc(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot AUROC comparison across models (basic version).

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
            colors = []

            for model in models:
                model_df = plot_df[plot_df["group_model"] == model]
                if len(model_df) > 0:
                    mean_val = model_df["false_belief_detection_auroc_mean"].iloc[0]
                    std_val = model_df["false_belief_detection_auroc_std"].iloc[0] if "false_belief_detection_auroc_std" in model_df.columns else 0.0

                    if mean_val is not None and not np.isnan(mean_val):
                        means.append(mean_val)
                        stds.append(std_val if std_val is not None else 0.0)
                        labels.append(model)
                        colors.append(self._get_model_color(model))

            if means:
                x_pos = np.arange(len(labels))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, fontweight='bold')
                ax.set_ylabel("AUROC", fontweight='bold')
                ax.set_title("False-Belief Detection AUROC by Model", fontweight='bold', fontsize=14)
                ax.set_ylim([0, 1.1])
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, "No AUROC data available", ha='center', va='center', fontsize=12)
            ax.set_title("False-Belief Detection AUROC")

        if save_path is None:
            save_path = self.output_dir / "detection_auroc.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_detection_auroc_detailed(
        self,
        save_path: Optional[Path] = None,
        show_individual_runs: bool = True,
        show_ci: bool = True,
    ) -> Path:
        """Plot detailed AUROC with individual runs, CI bands, and statistical annotations.

        Args:
            save_path: Path to save plot
            show_individual_runs: Whether to show individual run points
            show_ci: Whether to show confidence interval bands

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: Bar plot with individual points overlay
        ax1 = axes[0]
        if self.raw_df is not None and "model" in self.raw_df.columns:
            raw_fb = self.raw_df[self.raw_df["condition"] == "false_belief"].copy()
            models = ["reactive", "goal_only", "belief_pf"]
            x_positions = []

            for i, model in enumerate(models):
                model_data = raw_fb[raw_fb["model"] == model]["false_belief_detection_auroc"].dropna()
                if len(model_data) > 0:
                    # Bar for mean
                    mean_val = model_data.mean()
                    std_val = model_data.std()
                    color = self._get_model_color(model)

                    ax1.bar(i, mean_val, yerr=std_val, capsize=5, alpha=0.6,
                           color=color, edgecolor='black', linewidth=1, label=model if i == 0 else "")

                    # Individual points with jitter
                    if show_individual_runs:
                        jitter = np.random.normal(0, 0.05, len(model_data))
                        ax1.scatter(i + jitter, model_data.values, alpha=0.5, s=20,
                                   color=color, edgecolors='black', linewidth=0.5, zorder=5)

                    # 95% CI
                    if show_ci and len(model_data) > 1:
                        ci = stats.t.interval(0.95, len(model_data)-1, loc=mean_val, scale=stats.sem(model_data))
                        ax1.errorbar(i, mean_val, yerr=[[mean_val - ci[0]], [ci[1] - mean_val]],
                                    fmt='none', ecolor='black', capsize=7, capthick=2, zorder=6)

                    x_positions.append(i)

            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models, fontweight='bold')
            ax1.set_ylabel("AUROC", fontweight='bold')
            ax1.set_title("Detection AUROC with Individual Runs\n(False-Belief Condition)", fontweight='bold')
            ax1.set_ylim([0, 1.1])
            ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
            ax1.grid(True, alpha=0.3, axis='y')

        # Right: Violin plot showing distributions
        ax2 = axes[1]
        if self.raw_df is not None:
            raw_fb = self.raw_df[self.raw_df["condition"] == "false_belief"].copy()
            raw_fb = raw_fb[raw_fb["false_belief_detection_auroc"].notna()]

            if len(raw_fb) > 0:
                # Create violin plot
                models = raw_fb["model"].unique()
                data_list = []
                model_labels = []

                for model in ["reactive", "goal_only", "belief_pf"]:
                    if model in models:
                        model_data = raw_fb[raw_fb["model"] == model]["false_belief_detection_auroc"]
                        data_list.append(model_data.values)
                        model_labels.append(model)

                parts = ax2.violinplot(data_list, positions=range(len(data_list)), showmeans=True, showmedians=True)

                # Color violins
                for i, (pc, model) in enumerate(zip(parts['bodies'], model_labels)):
                    pc.set_facecolor(self._get_model_color(model))
                    pc.set_alpha(0.7)

                ax2.set_xticks(range(len(model_labels)))
                ax2.set_xticklabels(model_labels, fontweight='bold')
                ax2.set_ylabel("AUROC", fontweight='bold')
                ax2.set_title("Distribution of Detection AUROC\n(False-Belief Condition)", fontweight='bold')
                ax2.set_ylim([0, 1.1])
                ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax2.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = self.output_dir / "detection_auroc_detailed.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_detection_auroc_by_condition(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot AUROC grouped by condition with models side-by-side.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if self.raw_df is not None:
            conditions = ["control", "false_belief", "seen_relocation"]
            models = ["reactive", "goal_only", "belief_pf"]

            x = np.arange(len(conditions))
            width = 0.25

            for i, model in enumerate(models):
                means = []
                stds = []
                for cond in conditions:
                    subset = self.raw_df[(self.raw_df["model"] == model) & (self.raw_df["condition"] == cond)]
                    auroc = subset["false_belief_detection_auroc"].dropna()
                    means.append(auroc.mean() if len(auroc) > 0 else 0)
                    stds.append(auroc.std() if len(auroc) > 1 else 0)

                ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                       label=model, color=self._get_model_color(model),
                       alpha=0.8, edgecolor='black', linewidth=1)

            ax.set_xlabel("Condition", fontweight='bold')
            ax.set_ylabel("AUROC", fontweight='bold')
            ax.set_title("Detection AUROC by Model and Condition", fontweight='bold', fontsize=14)
            ax.set_xticks(x + width)
            ax.set_xticklabels(conditions, fontweight='bold')
            ax.legend(title="Model", loc='upper right')
            ax.set_ylim([0, 1.1])
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = self.output_dir / "detection_auroc_by_condition.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    # =========================================================================
    # DETECTION LATENCY ANALYSIS
    # =========================================================================

    def plot_detection_latency_histogram(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot histogram of detection delays by model.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        if self.raw_df is not None:
            models = ["reactive", "goal_only", "belief_pf"]

            for i, model in enumerate(models):
                ax = axes[i]
                model_data = self.raw_df[
                    (self.raw_df["model"] == model) &
                    (self.raw_df["condition"] == "false_belief")
                ]["false_belief_detection_latency"].dropna()

                if len(model_data) > 0:
                    ax.hist(model_data, bins=20, alpha=0.7,
                           color=self._get_model_color(model), edgecolor='black')
                    ax.axvline(model_data.mean(), color='red', linestyle='--',
                              linewidth=2, label=f'Mean: {model_data.mean():.1f}')
                    ax.axvline(model_data.median(), color='blue', linestyle=':',
                              linewidth=2, label=f'Median: {model_data.median():.1f}')
                    ax.legend(fontsize=8)

                ax.set_xlabel("Detection Latency (steps)", fontweight='bold')
                ax.set_title(f"{model}", fontweight='bold', fontsize=12)
                ax.grid(True, alpha=0.3)

            axes[0].set_ylabel("Frequency", fontweight='bold')

        fig.suptitle("Distribution of Detection Latency by Model", fontweight='bold', fontsize=14, y=1.02)

        if save_path is None:
            save_path = self.output_dir / "detection_latency_histogram.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_detection_latency_cdf(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot cumulative distribution function of detection latency.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if self.raw_df is not None:
            models = ["reactive", "goal_only", "belief_pf"]

            for model in models:
                model_data = self.raw_df[
                    (self.raw_df["model"] == model) &
                    (self.raw_df["condition"] == "false_belief")
                ]["false_belief_detection_latency"].dropna()

                if len(model_data) > 0:
                    sorted_data = np.sort(model_data)
                    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                    ax.plot(sorted_data, cdf, linewidth=2,
                           color=self._get_model_color(model), label=model)

            ax.set_xlabel("Detection Latency (steps)", fontweight='bold')
            ax.set_ylabel("Cumulative Probability", fontweight='bold')
            ax.set_title("CDF of Detection Latency", fontweight='bold', fontsize=14)
            ax.legend(title="Model", loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])

        if save_path is None:
            save_path = self.output_dir / "detection_latency_cdf.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_detection_latency_boxplot(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot box plots of detection latency by model and condition.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if self.raw_df is not None:
            # By model
            ax1 = axes[0]
            models = ["reactive", "goal_only", "belief_pf"]
            data = []
            labels = []
            colors = []

            for model in models:
                model_data = self.raw_df[
                    (self.raw_df["model"] == model) &
                    (self.raw_df["condition"] == "false_belief")
                ]["false_belief_detection_latency"].dropna()

                if len(model_data) > 0:
                    data.append(model_data.values)
                    labels.append(model)
                    colors.append(self._get_model_color(model))

            if data:
                bp = ax1.boxplot(data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax1.set_ylabel("Detection Latency (steps)", fontweight='bold')
                ax1.set_title("Detection Latency by Model", fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='y')

            # By condition
            ax2 = axes[1]
            conditions = ["control", "false_belief", "seen_relocation"]
            data = []
            labels = []
            colors = []

            for cond in conditions:
                cond_data = self.raw_df[self.raw_df["condition"] == cond]["false_belief_detection_latency"].dropna()
                if len(cond_data) > 0:
                    data.append(cond_data.values)
                    labels.append(cond)
                    colors.append(self._get_condition_color(cond))

            if data:
                bp = ax2.boxplot(data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax2.set_ylabel("Detection Latency (steps)", fontweight='bold')
                ax2.set_title("Detection Latency by Condition", fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = self.output_dir / "detection_latency_boxplot.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    # =========================================================================
    # TASK PERFORMANCE COMPARISONS
    # =========================================================================

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
            colors = []

            for model in models:
                model_df = self.df[self.df["group_model"] == model]
                if len(model_df) > 0 and "task_completed_mean" in model_df.columns:
                    rate = model_df["task_completed_mean"].iloc[0]
                    if rate is not None and not np.isnan(rate):
                        completion_rates.append(rate)
                        labels.append(model)
                        colors.append(self._get_model_color(model))

            if completion_rates:
                x_pos = np.arange(len(labels))
                ax1.bar(x_pos, completion_rates, alpha=0.8, color=colors, edgecolor='black', linewidth=1)
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(labels, fontweight='bold')
                ax1.set_ylabel("Completion Rate", fontweight='bold')
                ax1.set_title("Task Completion Rate by Model", fontweight='bold')
                ax1.set_ylim([0, 1.1])
                ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Wasted actions
        ax2 = axes[1]
        if "group_model" in self.df.columns:
            models = self.df["group_model"].unique()
            wasted_means = []
            wasted_stds = []
            labels = []
            colors = []

            for model in models:
                model_df = self.df[self.df["group_model"] == model]
                if len(model_df) > 0:
                    mean_val = model_df["num_wasted_actions_mean"].iloc[0] if "num_wasted_actions_mean" in model_df.columns else None
                    std_val = model_df["num_wasted_actions_std"].iloc[0] if "num_wasted_actions_std" in model_df.columns else None

                    if mean_val is not None and not np.isnan(mean_val):
                        wasted_means.append(mean_val)
                        wasted_stds.append(std_val if std_val is not None else 0.0)
                        labels.append(model)
                        colors.append(self._get_model_color(model))

            if wasted_means:
                x_pos = np.arange(len(labels))
                ax2.bar(x_pos, wasted_means, yerr=wasted_stds, capsize=5, alpha=0.8,
                       color=colors, edgecolor='black', linewidth=1)
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(labels, fontweight='bold')
                ax2.set_ylabel("Wasted Actions", fontweight='bold')
                ax2.set_title("Average Wasted Actions by Model", fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = self.output_dir / "task_performance.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_task_performance_detailed(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot detailed task performance with violin plots.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        if self.raw_df is not None:
            models = ["reactive", "goal_only", "belief_pf"]

            # 1. Task efficiency violin plot
            ax1 = axes[0, 0]
            data_list = []
            labels = []
            for model in models:
                eff = self.raw_df[self.raw_df["model"] == model]["task_efficiency"].dropna()
                if len(eff) > 0:
                    data_list.append(eff.values)
                    labels.append(model)

            if data_list:
                parts = ax1.violinplot(data_list, showmeans=True, showmedians=True)
                for i, (pc, model) in enumerate(zip(parts['bodies'], labels)):
                    pc.set_facecolor(self._get_model_color(model))
                    pc.set_alpha(0.7)
                ax1.set_xticks(range(1, len(labels) + 1))
                ax1.set_xticklabels(labels, fontweight='bold')
                ax1.set_ylabel("Task Efficiency", fontweight='bold')
                ax1.set_title("Task Efficiency Distribution", fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='y')

            # 2. Wasted actions by condition
            ax2 = axes[0, 1]
            conditions = ["control", "false_belief", "seen_relocation"]
            x = np.arange(len(conditions))
            width = 0.25

            for i, model in enumerate(models):
                means = []
                stds = []
                for cond in conditions:
                    subset = self.raw_df[(self.raw_df["model"] == model) & (self.raw_df["condition"] == cond)]
                    wa = subset["num_wasted_actions"]
                    means.append(wa.mean() if len(wa) > 0 else 0)
                    stds.append(wa.std() if len(wa) > 1 else 0)

                ax2.bar(x + i * width, means, width, yerr=stds, capsize=3,
                       label=model, color=self._get_model_color(model),
                       alpha=0.8, edgecolor='black')

            ax2.set_xlabel("Condition", fontweight='bold')
            ax2.set_ylabel("Wasted Actions", fontweight='bold')
            ax2.set_title("Wasted Actions by Model and Condition", fontweight='bold')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(conditions)
            ax2.legend(title="Model")
            ax2.grid(True, alpha=0.3, axis='y')

            # 3. Helper actions by model
            ax3 = axes[1, 0]
            data_list = []
            labels = []
            for model in models:
                ha = self.raw_df[self.raw_df["model"] == model]["num_helper_actions"].dropna()
                if len(ha) > 0:
                    data_list.append(ha.values)
                    labels.append(model)

            if data_list:
                bp = ax3.boxplot(data_list, labels=labels, patch_artist=True)
                for patch, label in zip(bp['boxes'], labels):
                    patch.set_facecolor(self._get_model_color(label))
                    patch.set_alpha(0.7)
                ax3.set_ylabel("Helper Actions", fontweight='bold')
                ax3.set_title("Helper Actions Distribution", fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='y')

            # 4. Intervention count by model
            ax4 = axes[1, 1]
            data_list = []
            labels = []
            for model in models:
                interventions = self.raw_df[self.raw_df["model"] == model]["num_interventions"].dropna()
                if len(interventions) > 0:
                    data_list.append(interventions.values)
                    labels.append(model)

            if data_list:
                bp = ax4.boxplot(data_list, labels=labels, patch_artist=True)
                for patch, label in zip(bp['boxes'], labels):
                    patch.set_facecolor(self._get_model_color(label))
                    patch.set_alpha(0.7)
                ax4.set_ylabel("Number of Interventions", fontweight='bold')
                ax4.set_title("Intervention Count Distribution", fontweight='bold')
                ax4.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = self.output_dir / "task_performance_detailed.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    # =========================================================================
    # INTERVENTION QUALITY ANALYSIS
    # =========================================================================

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
                ax1.bar(x_pos - width/2, precision_means, width, label='Precision',
                       alpha=0.8, color='#3498DB', edgecolor='black')
                ax1.bar(x_pos + width/2, recall_means, width, label='Recall',
                       alpha=0.8, color='#E74C3C', edgecolor='black')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(labels, fontweight='bold')
                ax1.set_ylabel("Score", fontweight='bold')
                ax1.set_title("Intervention Precision/Recall by Model", fontweight='bold')
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
                ax2.bar(x_pos - width/2, over_means, width, label='Over-correction',
                       alpha=0.8, color='#E67E22', edgecolor='black')
                ax2.bar(x_pos + width/2, under_means, width, label='Under-correction',
                       alpha=0.8, color='#9B59B6', edgecolor='black')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(labels, fontweight='bold')
                ax2.set_ylabel("Count", fontweight='bold')
                ax2.set_title("Over/Under-Correction by Model", fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = self.output_dir / "intervention_quality.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_intervention_precision_recall_scatter(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot precision vs recall scatter for each run.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if self.raw_df is not None:
            models = ["reactive", "goal_only", "belief_pf"]

            for model in models:
                model_data = self.raw_df[self.raw_df["model"] == model]
                prec = model_data["intervention_precision"]
                rec = model_data["intervention_recall"]

                ax.scatter(prec, rec, label=model, alpha=0.6, s=50,
                          color=self._get_model_color(model), edgecolors='black', linewidth=0.5)

            # Add diagonal line (F1 = 0.5 iso-line)
            x = np.linspace(0, 1, 100)
            for f1 in [0.2, 0.4, 0.6, 0.8]:
                y = f1 * x / (2 * x - f1)
                valid = (y >= 0) & (y <= 1)
                ax.plot(x[valid], y[valid], 'k--', alpha=0.3, linewidth=1)

            ax.set_xlabel("Precision", fontweight='bold')
            ax.set_ylabel("Recall", fontweight='bold')
            ax.set_title("Intervention Precision vs Recall", fontweight='bold', fontsize=14)
            ax.legend(title="Model", loc='lower left')
            ax.set_xlim([0, 1.05])
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)

        if save_path is None:
            save_path = self.output_dir / "intervention_pr_scatter.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_intervention_timing_distribution(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot distribution of intervention counts by model and condition.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        if self.raw_df is not None:
            models = ["reactive", "goal_only", "belief_pf"]

            for i, model in enumerate(models):
                ax = axes[i]
                model_data = self.raw_df[self.raw_df["model"] == model]

                # Separate by condition
                for cond in ["control", "false_belief", "seen_relocation"]:
                    cond_data = model_data[model_data["condition"] == cond]["num_interventions"]
                    ax.hist(cond_data, bins=15, alpha=0.5,
                           label=cond, color=self._get_condition_color(cond))

                ax.set_xlabel("Number of Interventions", fontweight='bold')
                ax.set_title(f"{model}", fontweight='bold', fontsize=12)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            axes[0].set_ylabel("Frequency", fontweight='bold')

        fig.suptitle("Intervention Count Distribution by Model", fontweight='bold', fontsize=14, y=1.02)

        if save_path is None:
            save_path = self.output_dir / "intervention_timing_dist.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    # =========================================================================
    # BELIEF TRACKING VISUALIZATIONS
    # =========================================================================

    def plot_belief_timeline(
        self,
        episode_id: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot belief state timeline for an episode.

        Args:
            episode_id: Episode ID to plot (if None, plots aggregated)
            save_path: Path to save plot (if None, uses default)

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot goal inference accuracy by model
        if "goal_inference_accuracy_mean" in self.df.columns and "group_model" in self.df.columns:
            models = self.df["group_model"].unique()
            for model in models:
                model_df = self.df[self.df["group_model"] == model]
                if len(model_df) > 0:
                    acc = model_df["goal_inference_accuracy_mean"].iloc[0]
                    if acc is not None and not np.isnan(acc):
                        ax.bar(model, acc, color=self._get_model_color(model),
                              alpha=0.8, edgecolor='black', linewidth=1)

        ax.set_xlabel("Model", fontweight='bold')
        ax.set_ylabel("Goal Inference Accuracy", fontweight='bold')
        ax.set_title("Goal Inference Accuracy by Model", fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = self.output_dir / "belief_timeline.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_goal_inference_by_condition(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot goal inference accuracy by model and condition.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if self.raw_df is not None:
            conditions = ["control", "false_belief", "seen_relocation"]
            models = ["reactive", "goal_only", "belief_pf"]

            x = np.arange(len(conditions))
            width = 0.25

            for i, model in enumerate(models):
                means = []
                stds = []
                for cond in conditions:
                    subset = self.raw_df[(self.raw_df["model"] == model) & (self.raw_df["condition"] == cond)]
                    acc = subset["goal_inference_accuracy"].dropna()
                    means.append(acc.mean() if len(acc) > 0 else 0)
                    stds.append(acc.std() if len(acc) > 1 else 0)

                ax.bar(x + i * width, means, width, yerr=stds, capsize=3,
                       label=model, color=self._get_model_color(model),
                       alpha=0.8, edgecolor='black')

            ax.set_xlabel("Condition", fontweight='bold')
            ax.set_ylabel("Goal Inference Accuracy", fontweight='bold')
            ax.set_title("Goal Inference Accuracy by Model and Condition", fontweight='bold', fontsize=14)
            ax.set_xticks(x + width)
            ax.set_xticklabels(conditions, fontweight='bold')
            ax.legend(title="Model", loc='upper right')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')

        if save_path is None:
            save_path = self.output_dir / "goal_inference_by_condition.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    # =========================================================================
    # COMPARISON HEATMAPS
    # =========================================================================

    def plot_model_comparison_heatmap(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot heatmap comparing models across all metrics.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        if self.raw_df is not None:
            models = ["reactive", "goal_only", "belief_pf"]
            metrics = [
                ("AUROC", "false_belief_detection_auroc"),
                ("Detection Latency", "false_belief_detection_latency"),
                ("Task Efficiency", "task_efficiency"),
                ("Wasted Actions", "num_wasted_actions"),
                ("Interventions", "num_interventions"),
                ("Precision", "intervention_precision"),
                ("Recall", "intervention_recall"),
                ("Over-corrections", "over_corrections"),
            ]

            # Build matrix
            matrix = []
            metric_labels = []

            for metric_name, metric_col in metrics:
                row = []
                for model in models:
                    val = self.raw_df[self.raw_df["model"] == model][metric_col].mean()
                    row.append(val if not np.isnan(val) else 0)
                matrix.append(row)
                metric_labels.append(metric_name)

            matrix = np.array(matrix)

            # Normalize each row to [0, 1] for visualization
            matrix_norm = np.zeros_like(matrix)
            for i in range(matrix.shape[0]):
                row_min, row_max = matrix[i].min(), matrix[i].max()
                if row_max - row_min > 0:
                    matrix_norm[i] = (matrix[i] - row_min) / (row_max - row_min)
                else:
                    matrix_norm[i] = 0.5

            # Create heatmap
            im = ax.imshow(matrix_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            # Labels
            ax.set_xticks(np.arange(len(models)))
            ax.set_yticks(np.arange(len(metric_labels)))
            ax.set_xticklabels(models, fontweight='bold')
            ax.set_yticklabels(metric_labels, fontweight='bold')

            # Add text annotations with actual values
            for i in range(len(metric_labels)):
                for j in range(len(models)):
                    val = matrix[i, j]
                    text = ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                  color='white' if matrix_norm[i, j] < 0.3 or matrix_norm[i, j] > 0.7 else 'black',
                                  fontsize=10, fontweight='bold')

            ax.set_title("Model Comparison Across Metrics\n(Green = Better, Red = Worse, normalized per metric)",
                        fontweight='bold', fontsize=14)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Normalized Score', fontweight='bold')

        if save_path is None:
            save_path = self.output_dir / "model_comparison_heatmap.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_condition_comparison_heatmap(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot heatmap comparing conditions across metrics.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        if self.raw_df is not None:
            conditions = ["control", "false_belief", "seen_relocation"]
            metrics = [
                ("AUROC", "false_belief_detection_auroc"),
                ("Detection Latency", "false_belief_detection_latency"),
                ("Task Efficiency", "task_efficiency"),
                ("Wasted Actions", "num_wasted_actions"),
                ("Interventions", "num_interventions"),
                ("Precision", "intervention_precision"),
                ("Recall", "intervention_recall"),
            ]

            matrix = []
            metric_labels = []

            for metric_name, metric_col in metrics:
                row = []
                for cond in conditions:
                    val = self.raw_df[self.raw_df["condition"] == cond][metric_col].mean()
                    row.append(val if not np.isnan(val) else 0)
                matrix.append(row)
                metric_labels.append(metric_name)

            matrix = np.array(matrix)

            # Normalize
            matrix_norm = np.zeros_like(matrix)
            for i in range(matrix.shape[0]):
                row_min, row_max = matrix[i].min(), matrix[i].max()
                if row_max - row_min > 0:
                    matrix_norm[i] = (matrix[i] - row_min) / (row_max - row_min)
                else:
                    matrix_norm[i] = 0.5

            im = ax.imshow(matrix_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

            ax.set_xticks(np.arange(len(conditions)))
            ax.set_yticks(np.arange(len(metric_labels)))
            ax.set_xticklabels(conditions, fontweight='bold')
            ax.set_yticklabels(metric_labels, fontweight='bold')

            for i in range(len(metric_labels)):
                for j in range(len(conditions)):
                    val = matrix[i, j]
                    text = ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                  color='white' if matrix_norm[i, j] < 0.3 or matrix_norm[i, j] > 0.7 else 'black',
                                  fontsize=10, fontweight='bold')

            ax.set_title("Condition Comparison Across Metrics\n(Green = Better, Red = Worse, normalized per metric)",
                        fontweight='bold', fontsize=14)

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Normalized Score', fontweight='bold')

        if save_path is None:
            save_path = self.output_dir / "condition_comparison_heatmap.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_statistical_significance_heatmap(
        self,
        metric: str = "false_belief_detection_auroc",
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot heatmap of p-values between all model pairs.

        Args:
            metric: Metric to compare
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        if self.raw_df is not None and metric in self.raw_df.columns:
            models = ["reactive", "goal_only", "belief_pf"]
            n = len(models)
            p_matrix = np.ones((n, n))

            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i != j:
                        data1 = self.raw_df[self.raw_df["model"] == model1][metric].dropna()
                        data2 = self.raw_df[self.raw_df["model"] == model2][metric].dropna()

                        if len(data1) > 1 and len(data2) > 1:
                            try:
                                _, p = stats.ttest_ind(data1, data2)
                                p_matrix[i, j] = p
                            except Exception:
                                p_matrix[i, j] = 1.0

            # Create heatmap
            im = ax.imshow(p_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.1)

            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            ax.set_xticklabels(models, fontweight='bold')
            ax.set_yticklabels(models, fontweight='bold')

            # Add p-value annotations
            for i in range(n):
                for j in range(n):
                    p = p_matrix[i, j]
                    if i == j:
                        text = "-"
                    elif p < 0.001:
                        text = f"p<0.001***"
                    elif p < 0.01:
                        text = f"{p:.3f}**"
                    elif p < 0.05:
                        text = f"{p:.3f}*"
                    else:
                        text = f"{p:.3f}"

                    ax.text(j, i, text, ha='center', va='center',
                           color='white' if p < 0.05 else 'black', fontsize=9)

            ax.set_title(f"Statistical Significance (t-test p-values)\nMetric: {metric}",
                        fontweight='bold', fontsize=12)

            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('p-value', fontweight='bold')

        if save_path is None:
            save_path = self.output_dir / f"significance_heatmap_{metric}.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    # =========================================================================
    # ABLATION STUDIES
    # =========================================================================

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
                ax.errorbar(param_values, metric_means, yerr=metric_stds,
                           marker='o', capsize=5, linewidth=2, markersize=8,
                           color='#3498DB', markeredgecolor='black')
                ax.fill_between(param_values,
                               np.array(metric_means) - np.array(metric_stds),
                               np.array(metric_means) + np.array(metric_stds),
                               alpha=0.2, color='#3498DB')
                ax.set_xlabel(parameter.replace('_', ' ').title(), fontweight='bold')
                ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
                ax.set_title(f"Ablation: {parameter} vs {metric}", fontweight='bold')
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
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    def plot_tau_effect(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Plot effect of tau (intervention threshold) on metrics.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        if self.raw_df is not None and "tau" in self.raw_df.columns:
            # Group by tau
            tau_values = sorted(self.raw_df["tau"].unique())

            # Left: AUROC by tau
            ax1 = axes[0]
            for model in ["reactive", "goal_only", "belief_pf"]:
                means = []
                stds = []
                for tau in tau_values:
                    subset = self.raw_df[(self.raw_df["model"] == model) & (self.raw_df["tau"] == tau)]
                    auroc = subset["false_belief_detection_auroc"].dropna()
                    means.append(auroc.mean() if len(auroc) > 0 else np.nan)
                    stds.append(auroc.std() if len(auroc) > 1 else 0)

                ax1.errorbar(tau_values, means, yerr=stds, marker='o', capsize=3,
                            label=model, color=self._get_model_color(model), linewidth=2)

            ax1.set_xlabel("Tau (Intervention Threshold)", fontweight='bold')
            ax1.set_ylabel("AUROC", fontweight='bold')
            ax1.set_title("Detection AUROC by Tau", fontweight='bold')
            ax1.legend(title="Model")
            ax1.grid(True, alpha=0.3)

            # Right: Wasted actions by tau
            ax2 = axes[1]
            for model in ["reactive", "goal_only", "belief_pf"]:
                means = []
                stds = []
                for tau in tau_values:
                    subset = self.raw_df[(self.raw_df["model"] == model) & (self.raw_df["tau"] == tau)]
                    wa = subset["num_wasted_actions"]
                    means.append(wa.mean() if len(wa) > 0 else np.nan)
                    stds.append(wa.std() if len(wa) > 1 else 0)

                ax2.errorbar(tau_values, means, yerr=stds, marker='o', capsize=3,
                            label=model, color=self._get_model_color(model), linewidth=2)

            ax2.set_xlabel("Tau (Intervention Threshold)", fontweight='bold')
            ax2.set_ylabel("Wasted Actions", fontweight='bold')
            ax2.set_title("Wasted Actions by Tau", fontweight='bold')
            ax2.legend(title="Model")
            ax2.grid(True, alpha=0.3)

        if save_path is None:
            save_path = self.output_dir / "tau_effect.png"
        else:
            save_path = Path(save_path)

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path

    # =========================================================================
    # COMBINED SUMMARY FIGURE
    # =========================================================================

    def plot_summary_figure(
        self,
        save_path: Optional[Path] = None,
    ) -> Path:
        """Create a comprehensive summary figure with multiple subplots.

        Args:
            save_path: Path to save plot

        Returns:
            Path to saved plot
        """
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        if self.raw_df is not None:
            models = ["reactive", "goal_only", "belief_pf"]
            conditions = ["control", "false_belief", "seen_relocation"]

            # 1. Detection AUROC bar chart
            ax1 = fig.add_subplot(gs[0, 0])
            for i, model in enumerate(models):
                data = self.raw_df[(self.raw_df["model"] == model) &
                                   (self.raw_df["condition"] == "false_belief")]
                auroc = data["false_belief_detection_auroc"].dropna()
                ax1.bar(i, auroc.mean() if len(auroc) > 0 else 0,
                       yerr=auroc.std() if len(auroc) > 1 else 0,
                       color=self._get_model_color(model), alpha=0.8,
                       edgecolor='black', capsize=3)
            ax1.set_xticks(range(len(models)))
            ax1.set_xticklabels(models)
            ax1.set_ylabel("AUROC")
            ax1.set_title("Detection AUROC\n(False-Belief)")
            ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax1.set_ylim([0, 1.1])
            ax1.grid(True, alpha=0.3, axis='y')

            # 2. Task efficiency box plot
            ax2 = fig.add_subplot(gs[0, 1])
            data_list = []
            for model in models:
                eff = self.raw_df[self.raw_df["model"] == model]["task_efficiency"].dropna()
                data_list.append(eff.values if len(eff) > 0 else [0])
            bp = ax2.boxplot(data_list, labels=models, patch_artist=True)
            for patch, model in zip(bp['boxes'], models):
                patch.set_facecolor(self._get_model_color(model))
                patch.set_alpha(0.7)
            ax2.set_ylabel("Task Efficiency")
            ax2.set_title("Task Efficiency\n(All Conditions)")
            ax2.grid(True, alpha=0.3, axis='y')

            # 3. Wasted actions by condition
            ax3 = fig.add_subplot(gs[0, 2])
            x = np.arange(len(conditions))
            width = 0.25
            for i, model in enumerate(models):
                means = []
                for cond in conditions:
                    subset = self.raw_df[(self.raw_df["model"] == model) &
                                        (self.raw_df["condition"] == cond)]
                    means.append(subset["num_wasted_actions"].mean())
                ax3.bar(x + i * width, means, width, label=model,
                       color=self._get_model_color(model), alpha=0.8, edgecolor='black')
            ax3.set_xticks(x + width)
            ax3.set_xticklabels(conditions, rotation=15, ha='right')
            ax3.set_ylabel("Wasted Actions")
            ax3.set_title("Wasted Actions\nby Condition")
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3, axis='y')

            # 4. Precision/Recall
            ax4 = fig.add_subplot(gs[1, 0])
            width = 0.35
            x_pos = np.arange(len(models))
            prec_means = [self.raw_df[self.raw_df["model"] == m]["intervention_precision"].mean() for m in models]
            rec_means = [self.raw_df[self.raw_df["model"] == m]["intervention_recall"].mean() for m in models]
            ax4.bar(x_pos - width/2, prec_means, width, label='Precision', color='#3498DB', alpha=0.8)
            ax4.bar(x_pos + width/2, rec_means, width, label='Recall', color='#E74C3C', alpha=0.8)
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(models)
            ax4.set_ylabel("Score")
            ax4.set_title("Intervention\nPrecision/Recall")
            ax4.legend(fontsize=8)
            ax4.set_ylim([0, 1.1])
            ax4.grid(True, alpha=0.3, axis='y')

            # 5. Interventions count
            ax5 = fig.add_subplot(gs[1, 1])
            data_list = []
            for model in models:
                data_list.append(self.raw_df[self.raw_df["model"] == model]["num_interventions"].values)
            bp = ax5.boxplot(data_list, labels=models, patch_artist=True)
            for patch, model in zip(bp['boxes'], models):
                patch.set_facecolor(self._get_model_color(model))
                patch.set_alpha(0.7)
            ax5.set_ylabel("Count")
            ax5.set_title("Number of\nInterventions")
            ax5.grid(True, alpha=0.3, axis='y')

            # 6. Over/Under-corrections
            ax6 = fig.add_subplot(gs[1, 2])
            width = 0.35
            over_means = [self.raw_df[self.raw_df["model"] == m]["over_corrections"].mean() for m in models]
            under_means = [self.raw_df[self.raw_df["model"] == m]["under_corrections"].mean() for m in models]
            ax6.bar(x_pos - width/2, over_means, width, label='Over', color='#E67E22', alpha=0.8)
            ax6.bar(x_pos + width/2, under_means, width, label='Under', color='#9B59B6', alpha=0.8)
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(models)
            ax6.set_ylabel("Count")
            ax6.set_title("Over/Under\nCorrections")
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3, axis='y')

            # 7-9. Large heatmap spanning bottom row
            ax7 = fig.add_subplot(gs[2, :])
            metrics = [
                ("AUROC", "false_belief_detection_auroc"),
                ("Efficiency", "task_efficiency"),
                ("Wasted", "num_wasted_actions"),
                ("Interventions", "num_interventions"),
                ("Precision", "intervention_precision"),
                ("Recall", "intervention_recall"),
            ]

            # Build matrix: rows = metrics, columns = model x condition
            col_labels = []
            matrix_data = []
            for metric_name, metric_col in metrics:
                row = []
                for model in models:
                    for cond in conditions:
                        if len(col_labels) < len(models) * len(conditions):
                            col_labels.append(f"{model[:4]}\n{cond[:4]}")
                        subset = self.raw_df[(self.raw_df["model"] == model) &
                                            (self.raw_df["condition"] == cond)]
                        val = subset[metric_col].mean() if metric_col in subset.columns else 0
                        row.append(val if not np.isnan(val) else 0)
                matrix_data.append(row)

            matrix = np.array(matrix_data)

            # Normalize per row
            matrix_norm = np.zeros_like(matrix)
            for i in range(matrix.shape[0]):
                row_min, row_max = matrix[i].min(), matrix[i].max()
                if row_max - row_min > 0:
                    matrix_norm[i] = (matrix[i] - row_min) / (row_max - row_min)
                else:
                    matrix_norm[i] = 0.5

            im = ax7.imshow(matrix_norm, cmap='RdYlGn', aspect='auto')
            ax7.set_xticks(np.arange(len(col_labels)))
            ax7.set_yticks(np.arange(len(metrics)))
            ax7.set_xticklabels(col_labels, fontsize=8)
            ax7.set_yticklabels([m[0] for m in metrics])

            # Add text
            for i in range(len(metrics)):
                for j in range(len(col_labels)):
                    val = matrix[i, j]
                    ax7.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if matrix_norm[i, j] < 0.3 or matrix_norm[i, j] > 0.7 else 'black')

            ax7.set_title("Complete Metrics Summary (Model x Condition)", fontweight='bold')

        fig.suptitle("Belief-State Assistance Research - Results Summary",
                    fontweight='bold', fontsize=16, y=0.98)

        if save_path is None:
            save_path = self.output_dir / "summary_figure.png"
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close()

        return save_path


def generate_plots(
    config: Dict[str, Any],
    aggregated_df: Optional[pd.DataFrame] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> List[Path]:
    """Generate plots from config (called by analysis pipeline).

    Args:
        config: Analysis configuration dictionary
        aggregated_df: Aggregated DataFrame (if None, loads from config)
        raw_df: Raw DataFrame (if None, loads from config)

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
        raw_df = df

    # Create plot generator
    plotter = PlotGenerator(aggregated_df, output_dir, raw_df=raw_df)

    # Get plot specifications from config
    plots_config = config.get("plots", [])

    generated_plots = []

    # Generate each plot from config
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


def generate_all_plots(
    config: Dict[str, Any],
    aggregated_df: Optional[pd.DataFrame] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> List[Path]:
    """Generate all comprehensive plots (new entry point for detailed visualization).

    Args:
        config: Analysis configuration dictionary
        aggregated_df: Aggregated DataFrame
        raw_df: Raw DataFrame for detailed plots

    Returns:
        List of paths to all generated plots
    """
    # Get output directory from config
    analysis_config = config.get("analysis", {})
    output_dir = Path(analysis_config.get("output_dir", "results/figures"))

    # Load results if not provided
    if aggregated_df is None or raw_df is None:
        from ..analysis.aggregate import AnalysisAggregator
        aggregator = AnalysisAggregator()
        input_dir = Path(analysis_config.get("input_dir", "results/metrics"))
        raw_df = aggregator.load_results(input_dir=input_dir)
        aggregated_df = aggregator.aggregate_metrics(raw_df)

    # Create plot generator
    plotter = PlotGenerator(aggregated_df, output_dir, raw_df=raw_df)

    generated_plots = []

    # Generate all plot types
    print("  Generating comprehensive visualization suite...")

    # 1. Detection AUROC plots
    print("    - Detection AUROC (basic)...")
    generated_plots.append(plotter.plot_detection_auroc())
    print("    - Detection AUROC (detailed with individual runs)...")
    generated_plots.append(plotter.plot_detection_auroc_detailed())
    print("    - Detection AUROC by condition...")
    generated_plots.append(plotter.plot_detection_auroc_by_condition())

    # 2. Detection latency plots
    print("    - Detection latency histogram...")
    generated_plots.append(plotter.plot_detection_latency_histogram())
    print("    - Detection latency CDF...")
    generated_plots.append(plotter.plot_detection_latency_cdf())
    print("    - Detection latency boxplot...")
    generated_plots.append(plotter.plot_detection_latency_boxplot())

    # 3. Task performance plots
    print("    - Task performance (basic)...")
    generated_plots.append(plotter.plot_task_performance())
    print("    - Task performance (detailed)...")
    generated_plots.append(plotter.plot_task_performance_detailed())

    # 4. Intervention quality plots
    print("    - Intervention quality (basic)...")
    generated_plots.append(plotter.plot_intervention_quality())
    print("    - Intervention precision/recall scatter...")
    generated_plots.append(plotter.plot_intervention_precision_recall_scatter())
    print("    - Intervention timing distribution...")
    generated_plots.append(plotter.plot_intervention_timing_distribution())

    # 5. Belief tracking plots
    print("    - Belief timeline...")
    generated_plots.append(plotter.plot_belief_timeline())
    print("    - Goal inference by condition...")
    generated_plots.append(plotter.plot_goal_inference_by_condition())

    # 6. Comparison heatmaps
    print("    - Model comparison heatmap...")
    generated_plots.append(plotter.plot_model_comparison_heatmap())
    print("    - Condition comparison heatmap...")
    generated_plots.append(plotter.plot_condition_comparison_heatmap())
    print("    - Statistical significance heatmap (AUROC)...")
    generated_plots.append(plotter.plot_statistical_significance_heatmap(metric="false_belief_detection_auroc"))

    # 7. Ablation plots
    print("    - Tau effect analysis...")
    generated_plots.append(plotter.plot_tau_effect())

    # 8. Summary figure
    print("    - Comprehensive summary figure...")
    generated_plots.append(plotter.plot_summary_figure())

    # Filter out None values
    generated_plots = [p for p in generated_plots if p is not None]

    print(f"  [OK] Generated {len(generated_plots)} plots")

    return generated_plots
