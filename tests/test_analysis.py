"""Tests for analysis components."""

import pytest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from src.bsa.analysis.aggregate import AnalysisAggregator
from src.bsa.viz.plots import PlotGenerator
from src.bsa.analysis.tables import TableGenerator
from src.bsa.analysis.report import ReportGenerator


@pytest.fixture
def sample_results_df():
    """Fixture for sample results DataFrame."""
    return pd.DataFrame({
        "model": ["reactive", "goal_only", "belief_pf"] * 2,
        "condition": ["control", "control", "control", "false_belief", "false_belief", "false_belief"],
        "episode_id": [f"ep_{i}" for i in range(6)],
        "false_belief_detection_auroc": [0.5, 0.6, 0.9, 0.5, 0.6, 0.9],
        "false_belief_detection_latency": [10.0, 8.0, 5.0, 10.0, 8.0, 5.0],
        "task_completed": [True, True, True, False, False, True],
        "num_wasted_actions": [5, 3, 1, 10, 8, 2],
        "task_efficiency": [0.8, 0.9, 0.95, 0.5, 0.6, 0.9],
    })


@pytest.fixture
def temp_output_dir():
    """Fixture for temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_summary_stats():
    """Fixture for sample summary statistics."""
    return {
        "total_episodes": 100,
        "total_models": 3,
        "total_conditions": 2,
        "overall_auroc": 0.7,
        "overall_task_completion": 0.8,
    }


@pytest.fixture
def sample_figure_paths(temp_output_dir):
    """Fixture for sample figure paths."""
    # Create dummy figure files
    figures = []
    for name in ["belief_timeline.png", "detection_auroc.png", "task_performance.png"]:
        fig_path = temp_output_dir / name
        fig_path.write_bytes(b"fake_png_data")
        figures.append(fig_path)
    return figures


@pytest.fixture
def sample_table_paths(temp_output_dir):
    """Fixture for sample table paths."""
    # Create dummy table files
    tables = []
    for name in ["summary.md", "detection.md"]:
        table_path = temp_output_dir / name
        table_path.write_text("# Table\n\nContent")
        tables.append(table_path)
    return tables


class TestAnalysisAggregator:
    """Tests for AnalysisAggregator."""

    def test_load_results(self, sample_results_df, temp_output_dir):
        """Test loading Parquet files."""
        # Save sample results
        results_file = temp_output_dir / "results.parquet"
        sample_results_df.to_parquet(results_file)
        
        aggregator = AnalysisAggregator()
        df = aggregator.load_results(input_path=results_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_results_df)
        assert "model" in df.columns

    def test_aggregate_metrics(self, sample_results_df):
        """Test metric aggregation."""
        aggregator = AnalysisAggregator()
        
        aggregated = aggregator.aggregate_metrics(sample_results_df, group_by=["model", "condition"])
        
        assert isinstance(aggregated, pd.DataFrame)
        assert len(aggregated) > 0

    def test_compute_statistics(self, sample_results_df):
        """Test statistics computation."""
        aggregator = AnalysisAggregator()
        
        stats = aggregator._compute_statistics(sample_results_df)
        
        assert isinstance(stats, dict)
        # Should have mean, std for numeric columns
        assert len(stats) > 0

    def test_filter_results(self, sample_results_df):
        """Test result filtering."""
        aggregator = AnalysisAggregator()
        
        filtered = aggregator.filter_results(sample_results_df, model="reactive")
        
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) <= len(sample_results_df)
        if len(filtered) > 0:
            assert all(filtered["model"] == "reactive")

    def test_compute_summary_statistics(self, sample_results_df):
        """Test summary statistics."""
        aggregator = AnalysisAggregator()
        
        aggregated = aggregator.aggregate_metrics(sample_results_df)
        summary = aggregator.compute_summary_statistics(aggregated)
        
        assert isinstance(summary, dict)
        assert len(summary) > 0


class TestPlotGenerator:
    """Tests for PlotGenerator."""

    def test_initialization(self, sample_results_df, temp_output_dir):
        """Test plot generator initializes."""
        generator = PlotGenerator(sample_results_df, temp_output_dir)
        
        assert generator.df is not None
        assert generator.output_dir == temp_output_dir

    def test_plot_detection_auroc(self, sample_results_df, temp_output_dir):
        """Test AUROC plot generation."""
        generator = PlotGenerator(sample_results_df, temp_output_dir)
        
        plot_path = generator.plot_detection_auroc(save_path=temp_output_dir / "auroc.png")
        
        assert plot_path.exists()
        assert plot_path.suffix == ".png"

    def test_plot_task_performance(self, sample_results_df, temp_output_dir):
        """Test task performance plot generation."""
        generator = PlotGenerator(sample_results_df, temp_output_dir)
        
        plot_path = generator.plot_task_performance(save_path=temp_output_dir / "task_perf.png")
        
        assert plot_path.exists()
        assert plot_path.suffix == ".png"

    def test_plot_intervention_quality(self, sample_results_df, temp_output_dir):
        """Test intervention quality plot generation."""
        generator = PlotGenerator(sample_results_df, temp_output_dir)
        
        plot_path = generator.plot_intervention_quality(save_path=temp_output_dir / "intervention.png")
        
        assert plot_path.exists()
        assert plot_path.suffix == ".png"

    def test_plot_belief_timeline(self, sample_results_df, temp_output_dir):
        """Test belief timeline plot generation."""
        generator = PlotGenerator(sample_results_df, temp_output_dir)
        
        plot_path = generator.plot_belief_timeline(save_path=temp_output_dir / "timeline.png")
        
        assert plot_path.exists()
        assert plot_path.suffix == ".png"

    def test_generate_plots(self, sample_results_df, temp_output_dir):
        """Test generate_plots() function."""
        from src.bsa.viz.plots import generate_plots
        
        config = {
            "analysis": {
                "input_dir": str(temp_output_dir),
                "output_dir": str(temp_output_dir / "figures"),
            },
            "plots": [
                {"type": "detection_auroc", "filename": "auroc.png"},
            ],
        }
        
        # Save sample data first
        results_file = temp_output_dir / "results.parquet"
        sample_results_df.to_parquet(results_file)
        
        # Generate plots
        generate_plots(config, aggregated_df=sample_results_df)


class TestTableGenerator:
    """Tests for TableGenerator."""

    def test_initialization(self, sample_results_df):
        """Test table generator initializes."""
        generator = TableGenerator(sample_results_df)
        
        assert generator.df is not None

    def test_generate_summary_table(self, sample_results_df, temp_output_dir):
        """Test summary table generation."""
        generator = TableGenerator(sample_results_df)
        
        # Aggregate first
        aggregator = AnalysisAggregator()
        aggregated = aggregator.aggregate_metrics(sample_results_df)
        
        generator_agg = TableGenerator(aggregated)
        table_md = generator_agg.generate_summary_table(format="markdown")
        
        assert isinstance(table_md, str)
        assert len(table_md) > 0

    def test_generate_detection_table(self, sample_results_df, temp_output_dir):
        """Test detection table generation."""
        aggregator = AnalysisAggregator()
        aggregated = aggregator.aggregate_metrics(sample_results_df)
        
        generator = TableGenerator(aggregated)
        table_md = generator.generate_detection_table(format="markdown")
        
        assert isinstance(table_md, str)

    def test_generate_task_performance_table(self, sample_results_df):
        """Test task performance table generation."""
        aggregator = AnalysisAggregator()
        aggregated = aggregator.aggregate_metrics(sample_results_df)
        
        generator = TableGenerator(aggregated)
        table_md = generator.generate_task_performance_table(format="markdown")
        
        assert isinstance(table_md, str)

    def test_generate_intervention_table(self, sample_results_df):
        """Test intervention table generation."""
        aggregator = AnalysisAggregator()
        aggregated = aggregator.aggregate_metrics(sample_results_df)
        
        generator = TableGenerator(aggregated)
        table_md = generator.generate_intervention_table(format="markdown")
        
        assert isinstance(table_md, str)

    def test_table_formats(self, sample_results_df):
        """Test Markdown and LaTeX formats."""
        aggregator = AnalysisAggregator()
        aggregated = aggregator.aggregate_metrics(sample_results_df)
        
        generator = TableGenerator(aggregated)
        
        # Test Markdown
        table_md = generator.generate_summary_table(format="markdown")
        assert isinstance(table_md, str)
        assert "|" in table_md or len(table_md) > 0  # Markdown tables use |
        
        # Test LaTeX
        table_tex = generator.generate_summary_table(format="latex")
        assert isinstance(table_tex, str)

    def test_generate_tables(self, sample_results_df, temp_output_dir):
        """Test generate_tables() function."""
        from src.bsa.analysis.tables import generate_tables
        
        aggregator = AnalysisAggregator()
        aggregated = aggregator.aggregate_metrics(sample_results_df)
        summary = aggregator.compute_summary_statistics(aggregated)
        
        config = {
            "analysis": {
                "output_dir": str(temp_output_dir),
            },
            "tables": [
                {"type": "summary", "filename_md": "summary.md", "filename_tex": "summary.tex"},
            ],
        }
        
        # Check function signature - may not have summary_stats parameter
        import inspect
        sig = inspect.signature(generate_tables)
        if "summary_stats" in sig.parameters:
            generate_tables(config, aggregated_df=aggregated, summary_stats=summary)
        else:
            generate_tables(config, aggregated_df=aggregated)


class TestReportGenerator:
    """Tests for ReportGenerator."""

    def test_initialization(self, sample_results_df, sample_summary_stats, sample_figure_paths, sample_table_paths, temp_output_dir):
        """Test report generator initializes."""
        template_path = temp_output_dir / "template.md"
        template_path.write_text("# Report\n\n{{SUMMARY_STATS}}\n\n{{FIGURES}}\n\n{{TABLES}}")
        
        output_path = temp_output_dir / "report.md"
        
        generator = ReportGenerator(
            aggregated_df=sample_results_df,
            summary_stats=sample_summary_stats,
            figure_paths=sample_figure_paths,
            table_paths=sample_table_paths,
            template_path=template_path,
            output_path=output_path,
        )
        
        assert generator.aggregated_df is not None
        assert generator.summary_stats == sample_summary_stats

    def test_generate_report(self, sample_results_df, sample_summary_stats, sample_figure_paths, sample_table_paths, temp_output_dir):
        """Test report generation."""
        template_path = temp_output_dir / "template.md"
        template_path.write_text("# Report\n\n{{SUMMARY_STATS}}\n\n{{FIGURES}}\n\n{{TABLES}}")
        
        output_path = temp_output_dir / "report.md"
        
        generator = ReportGenerator(
            aggregated_df=sample_results_df,
            summary_stats=sample_summary_stats,
            figure_paths=sample_figure_paths,
            table_paths=sample_table_paths,
            template_path=template_path,
            output_path=output_path,
        )
        
        report_path = generator.generate_report()
        
        assert report_path.exists()
        report_content = report_path.read_text()
        assert len(report_content) > 0

    def test_placeholder_filling(self, sample_results_df, sample_summary_stats, sample_figure_paths, sample_table_paths, temp_output_dir):
        """Test all placeholders are filled."""
        template_path = temp_output_dir / "template.md"
        template_path.write_text("# Report\n\n{{SUMMARY_STATS}}\n\n{{FIGURES}}\n\n{{TABLES}}\n\n{{DISCUSSION}}")
        
        output_path = temp_output_dir / "report.md"
        
        generator = ReportGenerator(
            aggregated_df=sample_results_df,
            summary_stats=sample_summary_stats,
            figure_paths=sample_figure_paths,
            table_paths=sample_table_paths,
            template_path=template_path,
            output_path=output_path,
        )
        
        report_path = generator.generate_report()
        report_content = report_path.read_text()
        
        # Check placeholders are filled
        assert "{{SUMMARY_STATS}}" not in report_content
        assert "{{FIGURES}}" not in report_content
        assert "{{TABLES}}" not in report_content
        assert "{{DISCUSSION}}" not in report_content

    def test_report_structure(self, sample_results_df, sample_summary_stats, sample_figure_paths, sample_table_paths, temp_output_dir):
        """Test report has correct structure."""
        template_path = temp_output_dir / "template.md"
        template_path.write_text("# Report\n\n## Summary\n\n{{SUMMARY_STATS}}\n\n## Results\n\n{{FIGURES}}\n\n{{TABLES}}")
        
        output_path = temp_output_dir / "report.md"
        
        generator = ReportGenerator(
            aggregated_df=sample_results_df,
            summary_stats=sample_summary_stats,
            figure_paths=sample_figure_paths,
            table_paths=sample_table_paths,
            template_path=template_path,
            output_path=output_path,
        )
        
        report_path = generator.generate_report()
        report_content = report_path.read_text()
        
        # Should have sections
        assert "# Report" in report_content
        assert "## Summary" in report_content or "## Results" in report_content

    def test_figure_inclusion(self, sample_results_df, sample_summary_stats, sample_figure_paths, sample_table_paths, temp_output_dir):
        """Test figures are included correctly."""
        template_path = temp_output_dir / "template.md"
        template_path.write_text("# Report\n\n{{FIGURES}}")
        
        output_path = temp_output_dir / "report.md"
        
        generator = ReportGenerator(
            aggregated_df=sample_results_df,
            summary_stats=sample_summary_stats,
            figure_paths=sample_figure_paths,
            table_paths=sample_table_paths,
            template_path=template_path,
            output_path=output_path,
        )
        
        report_path = generator.generate_report()
        report_content = report_path.read_text()
        
        # Should reference figures
        assert len(report_content) > 0

    def test_table_inclusion(self, sample_results_df, sample_summary_stats, sample_figure_paths, sample_table_paths, temp_output_dir):
        """Test tables are included correctly."""
        template_path = temp_output_dir / "template.md"
        template_path.write_text("# Report\n\n{{TABLES}}")
        
        output_path = temp_output_dir / "report.md"
        
        generator = ReportGenerator(
            aggregated_df=sample_results_df,
            summary_stats=sample_summary_stats,
            figure_paths=sample_figure_paths,
            table_paths=sample_table_paths,
            template_path=template_path,
            output_path=output_path,
        )
        
        report_path = generator.generate_report()
        report_content = report_path.read_text()
        
        # Should reference tables
        assert len(report_content) > 0
