"""Table generation module for summarizing experiment results.

# Fix: Comprehensive statistics table with CIs and sample sizes (Phase 10)
# Fix: Pairwise comparisons with effect sizes and significance tests (Phase 10)
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats

from .statistics import (
    effect_size,
    interpret_effect_size,
    compute_confidence_interval,
    compute_bootstrap_ci,
    paired_ttest,
    wilcoxon_test,
    independent_ttest,
    format_ci,
    format_p_value,
)


class TableGenerator:
    """Generator for creating publication-ready tables from experiment results."""

    def __init__(self, df: pd.DataFrame):
        """Initialize table generator.

        Args:
            df: Aggregated results DataFrame
        """
        self.df = df

    def generate_summary_table(
        self,
        format: str = "markdown",
    ) -> str:
        """Generate summary table comparing models across conditions.

        Args:
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string
        """
        # Filter for key metrics
        if "group_model" not in self.df.columns:
            return self._empty_table("Summary", format)
        
        # Get key metrics
        metrics = [
            ("AUROC", "false_belief_detection_auroc_mean", "false_belief_detection_auroc_std"),
            ("Detection Latency", "false_belief_detection_latency_mean", "false_belief_detection_latency_std"),
            ("Task Completion", "task_completed_mean", None),  # Will be converted to percentage
            ("Wasted Actions", "num_wasted_actions_mean", "num_wasted_actions_std"),
            ("Efficiency", "task_efficiency_mean", "task_efficiency_std"),
        ]
        
        # Build table rows
        rows = []
        for model in self.df["group_model"].unique():
            model_df = self.df[self.df["group_model"] == model]
            if len(model_df) == 0:
                continue
            
            row = {"Model": model}
            for metric_name, mean_col, std_col in metrics:
                if mean_col in model_df.columns:
                    mean_val = model_df[mean_col].iloc[0]
                    if mean_val is not None and not np.isnan(mean_val):
                        # Special handling for Task Completion (convert to percentage)
                        if metric_name == "Task Completion":
                            if std_col and std_col in model_df.columns:
                                std_val = model_df[std_col].iloc[0]
                                if std_val is not None and not np.isnan(std_val):
                                    # Convert to percentage
                                    row[metric_name] = f"{mean_val*100:.1f}% ± {std_val*100:.1f}%"
                                else:
                                    row[metric_name] = f"{mean_val*100:.1f}%"
                            else:
                                row[metric_name] = f"{mean_val*100:.1f}%"
                        else:
                            # Regular numeric formatting
                            if std_col and std_col in model_df.columns:
                                std_val = model_df[std_col].iloc[0]
                                if std_val is not None and not np.isnan(std_val):
                                    row[metric_name] = f"{mean_val:.3f} ± {std_val:.3f}"
                                else:
                                    row[metric_name] = f"{mean_val:.3f}"
                            else:
                                row[metric_name] = f"{mean_val:.3f}"
                    else:
                        row[metric_name] = "N/A"
                else:
                    row[metric_name] = "N/A"
            
            rows.append(row)
        
        if not rows:
            return self._empty_table("Summary", format)
        
        # Convert to DataFrame for formatting
        table_df = pd.DataFrame(rows)
        
        return self._format_table(table_df, "Summary Table", format)

    def generate_detection_table(
        self,
        format: str = "markdown",
    ) -> str:
        """Generate detection metrics table.

        Args:
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string
        """
        if "group_model" not in self.df.columns:
            return self._empty_table("Detection Metrics", format)
        
        # Filter for false_belief condition
        plot_df = self.df.copy()
        if "group_condition" in plot_df.columns:
            plot_df = plot_df[plot_df["group_condition"] == "false_belief"]
        
        rows = []
        for model in plot_df["group_model"].unique():
            model_df = plot_df[plot_df["group_model"] == model]
            if len(model_df) == 0:
                continue
            
            row = {"Model": model}
            
            # AUROC
            if "false_belief_detection_auroc_mean" in model_df.columns:
                mean_val = model_df["false_belief_detection_auroc_mean"].iloc[0]
                std_val = model_df["false_belief_detection_auroc_std"].iloc[0] if "false_belief_detection_auroc_std" in model_df.columns else None
                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        row["AUROC"] = f"{mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row["AUROC"] = f"{mean_val:.3f}"
                else:
                    row["AUROC"] = "N/A"
            else:
                row["AUROC"] = "N/A"
            
            # Detection Latency
            if "false_belief_detection_latency_mean" in model_df.columns:
                mean_val = model_df["false_belief_detection_latency_mean"].iloc[0]
                std_val = model_df["false_belief_detection_latency_std"].iloc[0] if "false_belief_detection_latency_std" in model_df.columns else None
                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        row["Detection Latency"] = f"{mean_val:.2f} ± {std_val:.2f}"
                    else:
                        row["Detection Latency"] = f"{mean_val:.2f}"
                else:
                    row["Detection Latency"] = "N/A"
            else:
                row["Detection Latency"] = "N/A"
            
            # FPR
            if "false_belief_detection_fpr_mean" in model_df.columns:
                mean_val = model_df["false_belief_detection_fpr_mean"].iloc[0]
                std_val = model_df["false_belief_detection_fpr_std"].iloc[0] if "false_belief_detection_fpr_std" in model_df.columns else None
                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        row["FPR"] = f"{mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row["FPR"] = f"{mean_val:.3f}"
                else:
                    row["FPR"] = "N/A"
            else:
                row["FPR"] = "N/A"
            
            rows.append(row)
        
        if not rows:
            return self._empty_table("Detection Metrics", format)
        
        table_df = pd.DataFrame(rows)
        return self._format_table(table_df, "False-Belief Detection Metrics", format)

    def generate_task_performance_table(
        self,
        format: str = "markdown",
    ) -> str:
        """Generate task performance table.

        Args:
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string
        """
        if "group_model" not in self.df.columns:
            return self._empty_table("Task Performance", format)
        
        rows = []
        for model in self.df["group_model"].unique():
            model_df = self.df[self.df["group_model"] == model]
            if len(model_df) == 0:
                continue
            
            row = {"Model": model}
            
            # Completion rate
            if "task_completed_mean" in model_df.columns:
                rate = model_df["task_completed_mean"].iloc[0]
                if rate is not None and not np.isnan(rate):
                    row["Completion Rate"] = f"{rate:.1%}"
                else:
                    row["Completion Rate"] = "N/A"
            else:
                row["Completion Rate"] = "N/A"
            
            # Steps
            if "num_steps_to_completion_mean" in model_df.columns:
                mean_val = model_df["num_steps_to_completion_mean"].iloc[0]
                std_val = model_df["num_steps_to_completion_std"].iloc[0] if "num_steps_to_completion_std" in model_df.columns else None
                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        row["Steps"] = f"{mean_val:.1f} ± {std_val:.1f}"
                    else:
                        row["Steps"] = f"{mean_val:.1f}"
                else:
                    row["Steps"] = "N/A"
            else:
                row["Steps"] = "N/A"
            
            # Wasted actions
            if "num_wasted_actions_mean" in model_df.columns:
                mean_val = model_df["num_wasted_actions_mean"].iloc[0]
                std_val = model_df["num_wasted_actions_std"].iloc[0] if "num_wasted_actions_std" in model_df.columns else None
                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        row["Wasted Actions"] = f"{mean_val:.1f} ± {std_val:.1f}"
                    else:
                        row["Wasted Actions"] = f"{mean_val:.1f}"
                else:
                    row["Wasted Actions"] = "N/A"
            else:
                row["Wasted Actions"] = "N/A"
            
            # Efficiency
            if "task_efficiency_mean" in model_df.columns:
                mean_val = model_df["task_efficiency_mean"].iloc[0]
                std_val = model_df["task_efficiency_std"].iloc[0] if "task_efficiency_std" in model_df.columns else None
                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        row["Efficiency"] = f"{mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row["Efficiency"] = f"{mean_val:.3f}"
                else:
                    row["Efficiency"] = "N/A"
            else:
                row["Efficiency"] = "N/A"
            
            rows.append(row)
        
        if not rows:
            return self._empty_table("Task Performance", format)
        
        table_df = pd.DataFrame(rows)
        return self._format_table(table_df, "Task Performance Metrics", format)

    def generate_intervention_table(
        self,
        format: str = "markdown",
    ) -> str:
        """Generate intervention quality table.

        Args:
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string
        """
        if "group_model" not in self.df.columns:
            return self._empty_table("Intervention Quality", format)
        
        rows = []
        for model in self.df["group_model"].unique():
            model_df = self.df[self.df["group_model"] == model]
            if len(model_df) == 0:
                continue
            
            row = {"Model": model}
            
            # Precision
            if "intervention_precision_mean" in model_df.columns:
                mean_val = model_df["intervention_precision_mean"].iloc[0]
                std_val = model_df["intervention_precision_std"].iloc[0] if "intervention_precision_std" in model_df.columns else None
                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        row["Precision"] = f"{mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row["Precision"] = f"{mean_val:.3f}"
                else:
                    row["Precision"] = "N/A"
            else:
                row["Precision"] = "N/A"
            
            # Recall
            if "intervention_recall_mean" in model_df.columns:
                mean_val = model_df["intervention_recall_mean"].iloc[0]
                std_val = model_df["intervention_recall_std"].iloc[0] if "intervention_recall_std" in model_df.columns else None
                if mean_val is not None and not np.isnan(mean_val):
                    if std_val is not None and not np.isnan(std_val):
                        row["Recall"] = f"{mean_val:.3f} ± {std_val:.3f}"
                    else:
                        row["Recall"] = f"{mean_val:.3f}"
                else:
                    row["Recall"] = "N/A"
            else:
                row["Recall"] = "N/A"
            
            # Over-corrections
            if "over_corrections_mean" in model_df.columns:
                mean_val = model_df["over_corrections_mean"].iloc[0]
                if mean_val is not None and not np.isnan(mean_val):
                    row["Over-corrections"] = f"{mean_val:.1f}"
                else:
                    row["Over-corrections"] = "N/A"
            else:
                row["Over-corrections"] = "N/A"
            
            # Under-corrections
            if "under_corrections_mean" in model_df.columns:
                mean_val = model_df["under_corrections_mean"].iloc[0]
                if mean_val is not None and not np.isnan(mean_val):
                    row["Under-corrections"] = f"{mean_val:.1f}"
                else:
                    row["Under-corrections"] = "N/A"
            else:
                row["Under-corrections"] = "N/A"
            
            rows.append(row)
        
        if not rows:
            return self._empty_table("Intervention Quality", format)
        
        table_df = pd.DataFrame(rows)
        return self._format_table(table_df, "Intervention Quality Metrics", format)

    def generate_ablation_table(
        self,
        parameter: str,
        format: str = "markdown",
    ) -> str:
        """Generate ablation table.

        Args:
            parameter: Parameter name to vary
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string or empty string if data not available
        """
        if parameter not in self.df.columns:
            return ""
        
        # Group by parameter value
        grouped = self.df.groupby(parameter)
        
        rows = []
        for param_val, group_df in grouped:
            row = {parameter: str(param_val)}
            
            # Add key metrics
            metrics = [
                ("AUROC", "false_belief_detection_auroc_mean"),
                ("Latency", "false_belief_detection_latency_mean"),
                ("Completion", "task_completed_mean"),
            ]
            
            for metric_name, col in metrics:
                if col in group_df.columns:
                    mean_val = group_df[col].mean()
                    if mean_val is not None and not np.isnan(mean_val):
                        row[metric_name] = f"{mean_val:.3f}"
                    else:
                        row[metric_name] = "N/A"
                else:
                    row[metric_name] = "N/A"
            
            rows.append(row)
        
        if not rows:
            return ""
        
        table_df = pd.DataFrame(rows)
        return self._format_table(table_df, f"Ablation: {parameter}", format)

    def _format_table(
        self,
        df: pd.DataFrame,
        title: str,
        format: str,
    ) -> str:
        """Format DataFrame as table.

        Args:
            df: DataFrame to format
            title: Table title
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string
        """
        if format == "latex":
            return self._format_latex(df, title)
        else:
            return self._format_markdown(df, title)

    def _format_markdown(
        self,
        df: pd.DataFrame,
        title: str,
    ) -> str:
        """Format DataFrame as Markdown table.

        Args:
            df: DataFrame to format
            title: Table title

        Returns:
            Markdown table string
        """
        lines = [f"## {title}", ""]
        
        # Header
        headers = list(df.columns)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # Rows
        for _, row in df.iterrows():
            values = [str(val) for val in row.values]
            lines.append("| " + " | ".join(values) + " |")
        
        lines.append("")
        return "\n".join(lines)

    def _format_latex(
        self,
        df: pd.DataFrame,
        title: str,
    ) -> str:
        """Format DataFrame as LaTeX table.

        Args:
            df: DataFrame to format
            title: Table title

        Returns:
            LaTeX table string
        """
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            f"\\caption{{{title}}}",
            "\\begin{tabular}{" + "c" * len(df.columns) + "}",
            "\\hline",
        ]
        
        # Header
        headers = [col.replace("_", "\\_") for col in df.columns]
        lines.append(" & ".join(headers) + " \\\\")
        lines.append("\\hline")
        
        # Rows
        for _, row in df.iterrows():
            values = [str(val).replace("_", "\\_") for val in row.values]
            lines.append(" & ".join(values) + " \\\\")
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
        
        return "\n".join(lines)

    def _empty_table(self, title: str, format: str) -> str:
        """Generate empty table placeholder.

        Args:
            title: Table title
            format: Output format

        Returns:
            Empty table string
        """
        if format == "latex":
            return f"\\begin{{table}}[h]\n\\centering\n\\caption{{{title}}}\n\\textit{{No data available}}\n\\end{{table}}\n"
        else:
            return f"## {title}\n\n*No data available*\n"

    def _compute_statistical_significance(
        self,
        model1_data: pd.Series,
        model2_data: pd.Series,
    ) -> tuple[float, str]:
        """Compute statistical significance between two models.

        Args:
            model1_data: Data for model 1
            model2_data: Data for model 2

        Returns:
            Tuple of (p-value, significance indicator)
        """
        # Remove NaN values
        data1 = model1_data.dropna()
        data2 = model2_data.dropna()
        
        if len(data1) < 2 or len(data2) < 2:
            return (1.0, "")
        
        # Use t-test for continuous data
        try:
            _, p_value = stats.ttest_ind(data1, data2)
            
            if p_value < 0.001:
                indicator = "***"
            elif p_value < 0.01:
                indicator = "**"
            elif p_value < 0.05:
                indicator = "*"
            else:
                indicator = ""
            
            return (p_value, indicator)
        except Exception:
            return (1.0, "")

    def generate_summary_statistics(
        self,
        raw_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        format: str = "markdown",
    ) -> str:
        """Generate comprehensive summary statistics table with CIs.

        # Fix: Comprehensive statistics table with CIs and sample sizes (Phase 10)

        Args:
            raw_df: Raw (non-aggregated) results DataFrame
            metrics: List of metrics to include (default: key metrics)
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string
        """
        if "model" not in raw_df.columns:
            return self._empty_table("Summary Statistics", format)
        
        if metrics is None:
            metrics = [
                ("AUROC", "auroc"),
                ("Detection Latency", "latency"),
                ("Efficiency", "efficiency"),
                ("Precision", "precision"),
                ("Recall", "recall"),
            ]
        
        rows = []
        for model in raw_df["model"].unique():
            model_df = raw_df[raw_df["model"] == model]
            
            row = {"Model": model}
            
            for metric_name, col in metrics:
                if col in model_df.columns:
                    values = model_df[col].dropna().values
                    if len(values) > 0:
                        ci_result = compute_bootstrap_ci(values, n_bootstrap=1000)
                        row[f"{metric_name}"] = format_ci(
                            ci_result["value"],
                            ci_result["ci_lower"],
                            ci_result["ci_upper"],
                        )
                        row[f"{metric_name} N"] = ci_result["n"]
                    else:
                        row[f"{metric_name}"] = "N/A"
                        row[f"{metric_name} N"] = 0
                else:
                    row[f"{metric_name}"] = "N/A"
                    row[f"{metric_name} N"] = 0
            
            rows.append(row)
        
        if not rows:
            return self._empty_table("Summary Statistics", format)
        
        table_df = pd.DataFrame(rows)
        return self._format_table(table_df, "Summary Statistics with 95% CI", format)

    def generate_pairwise_comparisons(
        self,
        raw_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        format: str = "markdown",
    ) -> str:
        """Generate pairwise comparison table with effect sizes and p-values.

        # Fix: Pairwise comparisons with effect sizes and significance tests (Phase 10)

        Args:
            raw_df: Raw (non-aggregated) results DataFrame
            metrics: List of metrics to compare (default: key metrics)
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string
        """
        if "model" not in raw_df.columns:
            return self._empty_table("Pairwise Comparisons", format)
        
        models = list(raw_df["model"].unique())
        if len(models) < 2:
            return self._empty_table("Pairwise Comparisons", format)
        
        if metrics is None:
            metrics = [
                ("AUROC", "auroc"),
                ("Efficiency", "efficiency"),
                ("Precision", "precision"),
            ]
        
        rows = []
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                df1 = raw_df[raw_df["model"] == model1]
                df2 = raw_df[raw_df["model"] == model2]
                
                for metric_name, col in metrics:
                    if col not in raw_df.columns:
                        continue
                    
                    values1 = df1[col].dropna().values
                    values2 = df2[col].dropna().values
                    
                    if len(values1) == 0 or len(values2) == 0:
                        continue
                    
                    # Effect size
                    d = effect_size(values1, values2)
                    d_interp = interpret_effect_size(d)
                    
                    # Statistical tests
                    ttest_result = independent_ttest(values1, values2)
                    
                    rows.append({
                        "Comparison": f"{model1} vs {model2}",
                        "Metric": metric_name,
                        "Mean 1": f"{np.mean(values1):.3f}",
                        "Mean 2": f"{np.mean(values2):.3f}",
                        "Cohen's d": f"{d:.3f}" if not np.isnan(d) else "N/A",
                        "Effect": d_interp,
                        "p-value": format_p_value(ttest_result["p_value"]),
                    })
        
        if not rows:
            return self._empty_table("Pairwise Comparisons", format)
        
        table_df = pd.DataFrame(rows)
        return self._format_table(table_df, "Pairwise Model Comparisons", format)

    def generate_condition_comparison(
        self,
        raw_df: pd.DataFrame,
        format: str = "markdown",
    ) -> str:
        """Generate condition comparison table.

        Args:
            raw_df: Raw (non-aggregated) results DataFrame
            format: Output format ('markdown' or 'latex')

        Returns:
            Formatted table string
        """
        if "condition" not in raw_df.columns:
            return self._empty_table("Condition Comparison", format)
        
        rows = []
        for condition in raw_df["condition"].unique():
            cond_df = raw_df[raw_df["condition"] == condition]
            
            row = {"Condition": condition, "N": len(cond_df)}
            
            # Key metrics
            for metric_name, col in [("AUROC", "auroc"), ("Efficiency", "efficiency")]:
                if col in cond_df.columns:
                    values = cond_df[col].dropna().values
                    if len(values) > 0:
                        ci_result = compute_bootstrap_ci(values, n_bootstrap=1000)
                        row[metric_name] = format_ci(
                            ci_result["value"],
                            ci_result["ci_lower"],
                            ci_result["ci_upper"],
                        )
                    else:
                        row[metric_name] = "N/A"
                else:
                    row[metric_name] = "N/A"
            
            rows.append(row)
        
        if not rows:
            return self._empty_table("Condition Comparison", format)
        
        table_df = pd.DataFrame(rows)
        return self._format_table(table_df, "Comparison Across Conditions", format)


def generate_tables(
    config: Dict[str, Any],
    aggregated_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Generate tables from config (called by analysis pipeline).

    Args:
        config: Analysis configuration dictionary
        aggregated_df: Aggregated DataFrame (if None, loads from config)
        output_dir: Output directory (if None, uses config or default)

    Returns:
        List of paths to generated table files
    """
    # Get output directory
    if output_dir is None:
        analysis_config = config.get("analysis", {})
        # Default to results/tables (not figures)
        output_dir = Path(analysis_config.get("tables_dir", "results/tables"))
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load aggregated results if not provided
    if aggregated_df is None:
        from ..analysis.aggregate import AnalysisAggregator
        aggregator = AnalysisAggregator()
        analysis_config = config.get("analysis", {})
        input_dir = Path(analysis_config.get("input_dir", "results/metrics"))
        df = aggregator.load_results(input_dir=input_dir)
        aggregated_df = aggregator.aggregate_metrics(df)
    
    # Create table generator
    table_gen = TableGenerator(aggregated_df)
    
    generated_tables = []
    
    # Generate all tables
    print("  Generating summary table...")
    summary_md = table_gen.generate_summary_table(format="markdown")
    summary_latex = table_gen.generate_summary_table(format="latex")
    
    summary_md_path = output_dir / "summary.md"
    summary_latex_path = output_dir / "summary.tex"
    
    summary_md_path.write_text(summary_md)
    summary_latex_path.write_text(summary_latex)
    generated_tables.extend([summary_md_path, summary_latex_path])
    print(f"    [OK] Saved to: {summary_md_path}, {summary_latex_path}")
    
    print("  Generating detection table...")
    detection_md = table_gen.generate_detection_table(format="markdown")
    detection_latex = table_gen.generate_detection_table(format="latex")
    
    detection_md_path = output_dir / "detection.md"
    detection_latex_path = output_dir / "detection.tex"
    
    detection_md_path.write_text(detection_md)
    detection_latex_path.write_text(detection_latex)
    generated_tables.extend([detection_md_path, detection_latex_path])
    print(f"    [OK] Saved to: {detection_md_path}, {detection_latex_path}")
    
    print("  Generating task performance table...")
    task_md = table_gen.generate_task_performance_table(format="markdown")
    task_latex = table_gen.generate_task_performance_table(format="latex")
    
    task_md_path = output_dir / "task_performance.md"
    task_latex_path = output_dir / "task_performance.tex"
    
    task_md_path.write_text(task_md)
    task_latex_path.write_text(task_latex)
    generated_tables.extend([task_md_path, task_latex_path])
    print(f"    [OK] Saved to: {task_md_path}, {task_latex_path}")
    
    print("  Generating intervention table...")
    intervention_md = table_gen.generate_intervention_table(format="markdown")
    intervention_latex = table_gen.generate_intervention_table(format="latex")
    
    intervention_md_path = output_dir / "intervention.md"
    intervention_latex_path = output_dir / "intervention.tex"
    
    intervention_md_path.write_text(intervention_md)
    intervention_latex_path.write_text(intervention_latex)
    generated_tables.extend([intervention_md_path, intervention_latex_path])
    print(f"    [OK] Saved to: {intervention_md_path}, {intervention_latex_path}")
    
    return generated_tables
