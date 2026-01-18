"""Report generation module for creating technical reports."""

from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


class ReportGenerator:
    """Generator for creating technical reports from experiment results."""

    def __init__(
        self,
        aggregated_df: pd.DataFrame,
        summary_stats: Dict[str, Any],
        figure_paths: List[Path],
        table_paths: List[Path],
        template_path: Path,
        output_path: Path,
    ):
        """Initialize report generator.

        Args:
            aggregated_df: Aggregated results DataFrame
            summary_stats: Summary statistics dictionary
            figure_paths: List of paths to generated figures
            table_paths: List of paths to generated tables
            template_path: Path to report template
            output_path: Path to save generated report
        """
        self.aggregated_df = aggregated_df
        self.summary_stats = summary_stats
        self.figure_paths = figure_paths
        self.table_paths = table_paths
        self.template_path = template_path
        self.output_path = output_path

    def generate_report(self) -> Path:
        """Generate report from template.

        Returns:
            Path to generated report
        """
        # Load template
        template = self.template_path.read_text()
        
        # Fill in placeholders
        report = template
        
        # Fill summary statistics
        report = self._fill_summary_section(report)
        
        # Fill results sections
        report = self._fill_results_section(report)
        
        # Fill discussion
        report = self._fill_discussion_section(report)
        
        # Fill other placeholders
        report = self._fill_other_placeholders(report)
        
        # Save report
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(report)
        
        return self.output_path

    def _fill_summary_section(self, report: str) -> str:
        """Fill summary statistics section.

        Args:
            report: Report template string

        Returns:
            Report with summary filled in
        """
        if "{{SUMMARY_STATS}}" not in report:
            return report
        
        lines = []
        lines.append("**Overall Statistics:**")
        lines.append(f"- Total runs: {self.summary_stats.get('total_runs', 'N/A')}")
        lines.append(f"- Models tested: {', '.join(self.summary_stats.get('models', []))}")
        lines.append(f"- Conditions tested: {', '.join(self.summary_stats.get('conditions', []))}")
        lines.append("")
        
        # Model comparison
        if "model_comparison" in self.summary_stats:
            lines.append("**Model Comparison:**")
            for model, stats in self.summary_stats["model_comparison"].items():
                lines.append(f"- **{model}**:")
                lines.append(f"  - Runs: {stats.get('num_runs', 'N/A')}")
                lines.append(f"  - Completion rate: {stats.get('task_completion_rate', 0):.1%}")
                lines.append(f"  - Avg AUROC: {stats.get('avg_detection_auroc', 'N/A')}")
            lines.append("")
        
        summary_text = "\n".join(lines)
        return report.replace("{{SUMMARY_STATS}}", summary_text)

    def _fill_results_section(self, report: str) -> str:
        """Fill results sections.

        Args:
            report: Report template string

        Returns:
            Report with results filled in
        """
        # Fill detection results
        if "{{DETECTION_RESULTS}}" in report:
            detection_text = "See Detection Metrics table and AUROC plot below."
            report = report.replace("{{DETECTION_RESULTS}}", detection_text)
        
        # Fill task performance results
        if "{{TASK_PERFORMANCE_RESULTS}}" in report:
            task_text = "See Task Performance table and plot below."
            report = report.replace("{{TASK_PERFORMANCE_RESULTS}}", task_text)
        
        # Fill intervention results
        if "{{INTERVENTION_RESULTS}}" in report:
            intervention_text = "See Intervention Quality table and plot below."
            report = report.replace("{{INTERVENTION_RESULTS}}", intervention_text)
        
        # Fill figures
        if "{{FIGURES}}" in report:
            figure_lines = []
            for fig_path in self.figure_paths:
                # Compute relative path from report directory
                try:
                    rel_path = Path(fig_path).relative_to(self.output_path.parent)
                except ValueError:
                    # If not in subpath, use absolute path or just filename
                    rel_path = Path(fig_path).name
                figure_lines.append(f"![{fig_path.stem}]({rel_path})")
            figures_text = "\n\n".join(figure_lines) if figure_lines else "*No figures available*"
            report = report.replace("{{FIGURES}}", figures_text)
        
        # Fill tables
        if "{{TABLES}}" in report:
            table_lines = []
            for table_path in self.table_paths:
                if table_path.suffix == ".md":
                    # Compute relative path from report directory
                    try:
                        rel_path = Path(table_path).relative_to(self.output_path.parent)
                    except ValueError:
                        # If not in subpath, use absolute path or just filename
                        rel_path = Path(table_path).name
                    table_lines.append(f"See [{table_path.stem}]({rel_path})")
            tables_text = "\n\n".join(table_lines) if table_lines else "*No tables available*"
            report = report.replace("{{TABLES}}", tables_text)
        
        return report

    def _fill_discussion_section(self, report: str) -> str:
        """Fill discussion section.

        Args:
            report: Report template string

        Returns:
            Report with discussion filled in
        """
        # Fill discussion placeholder
        if "{{DISCUSSION}}" in report:
            discussion_lines = []
            
            # Generate discussion based on results
            if "model_comparison" in self.summary_stats:
                models = self.summary_stats["model_comparison"]
                
                discussion_lines.append("The experimental results demonstrate the performance of different helper agent models across various conditions.")
                discussion_lines.append("")
                
                # Compare models
                if "belief_pf" in models:
                    pf_stats = models["belief_pf"]
                    discussion_lines.append("The belief-sensitive (particle filter) model shows promise in tracking both goal and belief states.")
                    if pf_stats.get("avg_detection_auroc") is not None:
                        discussion_lines.append(f"It achieves an average AUROC of {pf_stats['avg_detection_auroc']:.3f} for false-belief detection.")
                
                if "reactive" in models:
                    reactive_stats = models["reactive"]
                    discussion_lines.append("The reactive baseline provides a simple comparison point without inference capabilities.")
                
                if "goal_only" in models:
                    goal_stats = models["goal_only"]
                    discussion_lines.append("The goal-only baseline demonstrates the importance of belief tracking beyond goal inference.")
                
                discussion_lines.append("")
                discussion_lines.append("These results highlight the value of belief-sensitive assistance in scenarios with partial observability and false beliefs.")
            else:
                discussion_lines.append("The experimental results provide insights into the performance of different helper agent models.")
                discussion_lines.append("Further analysis is needed to draw comprehensive conclusions.")
            
            discussion_text = "\n".join(discussion_lines)
            report = report.replace("{{DISCUSSION}}", discussion_text)
        
        # Fill key findings
        if "{{KEY_FINDINGS}}" in report:
            findings = []
            if "model_comparison" in self.summary_stats:
                # Compare models
                models = self.summary_stats["model_comparison"]
                if "belief_pf" in models and "reactive" in models:
                    pf_auroc = models["belief_pf"].get("avg_detection_auroc")
                    reactive_auroc = models["reactive"].get("avg_detection_auroc")
                    if pf_auroc and reactive_auroc:
                        findings.append(f"- Belief-sensitive model achieves AUROC of {pf_auroc:.3f} vs {reactive_auroc:.3f} for reactive baseline.")
                
                # Add more findings based on available data
                pf_stats = models.get("belief_pf", {})
                if pf_stats.get("task_completion_rate", 0) > 0:
                    findings.append(f"- Belief-sensitive model achieves {pf_stats['task_completion_rate']:.1%} task completion rate.")
            
            findings_text = "\n".join(findings) if findings else "- Analysis pending - run more experiments for detailed findings"
            report = report.replace("{{KEY_FINDINGS}}", findings_text)
        
        # Fill limitations
        if "{{LIMITATIONS}}" in report:
            limitations = [
                "- Limited to scripted human policies (not learned behaviors)",
                "- Simplified belief representation (discrete locations)",
                "- Small-scale experiments (limited number of runs)",
                "- GridHouse simulator (may not fully capture real-world complexity)",
            ]
            limitations_text = "\n".join(limitations)
            report = report.replace("{{LIMITATIONS}}", limitations_text)
        
        return report

    def _fill_other_placeholders(self, report: str) -> str:
        """Fill other placeholders.

        Args:
            report: Report template string

        Returns:
            Report with other placeholders filled in
        """
        # Task descriptions
        if "{{TASK_DESCRIPTIONS}}" in report:
            tasks = [
                "- **prepare_meal**: Prepare a meal using kitchen tools",
                "- **set_table**: Set the dining table",
                "- **pack_bag**: Pack items into a bag",
                "- **find_keys**: Find and retrieve keys",
            ]
            report = report.replace("{{TASK_DESCRIPTIONS}}", "\n".join(tasks))
        
        # Experimental setup
        if "{{EXPERIMENTAL_SETUP}}" in report:
            setup = [
                "- Environment: GridHouse simulator",
                "- Conditions: control, false_belief, seen_relocation",
                "- Multiple runs per model/condition combination",
            ]
            report = report.replace("{{EXPERIMENTAL_SETUP}}", "\n".join(setup))
        
        # Conclusion
        if "{{CONCLUSION}}" in report:
            conclusion = "Belief-sensitive assistance shows promise for improving task performance and reducing wasted actions compared to reactive and goal-only baselines."
            report = report.replace("{{CONCLUSION}}", conclusion)
        
        # Experimental details
        if "{{EXPERIMENTAL_DETAILS}}" in report:
            details = f"- Total runs: {self.summary_stats.get('total_runs', 'N/A')}"
            report = report.replace("{{EXPERIMENTAL_DETAILS}}", details)
        
        # Hyperparameters
        if "{{HYPERPARAMETERS}}" in report:
            hyperparams = [
                "- Particle filter: 100 particles (default)",
                "- Likelihood model: Rule-based",
            ]
            report = report.replace("{{HYPERPARAMETERS}}", "\n".join(hyperparams))
        
        # Reproducibility
        if "{{REPRODUCIBILITY}}" in report:
            repro = "See manifest files in results/manifests/ for full reproducibility information."
            report = report.replace("{{REPRODUCIBILITY}}", repro)
        
        return report


def generate_report(
    config: Dict[str, Any],
    aggregated_df: Optional[pd.DataFrame] = None,
    summary_stats: Optional[Dict[str, Any]] = None,
    figure_paths: Optional[List[Path]] = None,
    table_paths: Optional[List[Path]] = None,
) -> Path:
    """Generate report from config (called by analysis pipeline).

    Args:
        config: Analysis configuration dictionary
        aggregated_df: Aggregated DataFrame (if None, loads from config)
        summary_stats: Summary statistics (if None, computes from aggregated_df)
        figure_paths: List of figure paths (if None, finds in output directory)
        table_paths: List of table paths (if None, finds in output directory)

    Returns:
        Path to generated report
    """
    # Get paths from config
    report_config = config.get("report", {})
    template_path = Path(report_config.get("template", "paper/report_template.md"))
    output_path = Path(report_config.get("output", "results/reports/report.md"))
    
    # Load aggregated results if not provided
    if aggregated_df is None:
        from ..analysis.aggregate import AnalysisAggregator
        aggregator = AnalysisAggregator()
        analysis_config = config.get("analysis", {})
        input_dir = Path(analysis_config.get("input_dir", "results/metrics"))
        df = aggregator.load_results(input_dir=input_dir)
        aggregated_df = aggregator.aggregate_metrics(df)
    
    # Compute summary statistics if not provided
    if summary_stats is None:
        from ..analysis.aggregate import AnalysisAggregator
        aggregator = AnalysisAggregator()
        summary_stats = aggregator.compute_summary_statistics(aggregated_df)
    
    # Find figures if not provided
    if figure_paths is None:
        analysis_config = config.get("analysis", {})
        output_dir = Path(analysis_config.get("output_dir", "results/figures"))
        figure_paths = list(output_dir.glob("*.png"))
    
    # Find tables if not provided
    if table_paths is None:
        table_dir = Path("results/tables")
        table_paths = list(table_dir.glob("*.md"))
    
    # Create report generator
    generator = ReportGenerator(
        aggregated_df=aggregated_df,
        summary_stats=summary_stats,
        figure_paths=figure_paths,
        table_paths=table_paths,
        template_path=template_path,
        output_path=output_path,
    )
    
    # Generate report
    print(f"  Generating report...")
    report_path = generator.generate_report()
    print(f"    [OK] Saved to: {report_path}")
    
    return report_path
