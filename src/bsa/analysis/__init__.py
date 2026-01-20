"""Analysis tools."""

from .aggregate import AnalysisAggregator, aggregate_results, analyze_results
from .tables import TableGenerator, generate_tables
from .report import ReportGenerator, generate_report
from .statistics import (
    compute_auroc_with_ci,
    compute_bootstrap_ci,
    compute_confidence_interval,
    effect_size,
    interpret_effect_size,
    paired_ttest,
    wilcoxon_test,
    independent_ttest,
    mann_whitney_test,
    format_ci,
    format_p_value,
    summary_statistics,
)
from .visualization import (
    plot_roc_curves,
    plot_auroc_distribution,
    plot_temporal_metrics,
    plot_efficiency_by_model,
    plot_condition_comparison,
    plot_model_comparison_heatmap,
    generate_all_figures,
)

__all__ = [
    "AnalysisAggregator",
    "aggregate_results",
    "analyze_results",
    "TableGenerator",
    "generate_tables",
    "ReportGenerator",
    "generate_report",
    # Statistics exports (Phase 10)
    "compute_auroc_with_ci",
    "compute_bootstrap_ci",
    "compute_confidence_interval",
    "effect_size",
    "interpret_effect_size",
    "paired_ttest",
    "wilcoxon_test",
    "independent_ttest",
    "mann_whitney_test",
    "format_ci",
    "format_p_value",
    "summary_statistics",
    # Visualization exports (Phase 10)
    "plot_roc_curves",
    "plot_auroc_distribution",
    "plot_temporal_metrics",
    "plot_efficiency_by_model",
    "plot_condition_comparison",
    "plot_model_comparison_heatmap",
    "generate_all_figures",
]
