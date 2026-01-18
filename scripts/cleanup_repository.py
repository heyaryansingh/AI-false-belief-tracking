#!/usr/bin/env python3
"""Clean up repository by removing unnecessary files."""

from pathlib import Path
import shutil

def cleanup():
    """Remove unnecessary files."""
    removed = []
    
    print("=" * 70)
    print("Repository Cleanup")
    print("=" * 70)
    
    # 1. Remove backup/duplicate paper files
    print("\n[1] Removing backup/duplicate paper files...")
    backup_files = [
        "paper/research_paper_original.md",
        "paper/research_paper_enhanced.md",
    ]
    for f in backup_files:
        p = Path(f)
        if p.exists():
            p.unlink()
            removed.append(f)
            print(f"  Removed: {f}")
    
    # 2. Remove duplicate figures from results/ root (keep only in results/figures/)
    print("\n[2] Removing duplicate figures from results/ root...")
    figure_files = [
        "results/belief_timeline.png",
        "results/condition_comparison_heatmap.png",
        "results/detection_auroc_by_condition.png",
        "results/detection_auroc_detailed.png",
        "results/detection_auroc.png",
        "results/detection_latency_boxplot.png",
        "results/detection_latency_cdf.png",
        "results/detection_latency_histogram.png",
        "results/goal_inference_by_condition.png",
        "results/intervention_pr_scatter.png",
        "results/intervention_quality.png",
        "results/intervention_timing_dist.png",
        "results/model_comparison_heatmap.png",
        "results/significance_heatmap_false_belief_detection_auroc.png",
        "results/summary_figure.png",
        "results/task_performance_detailed.png",
        "results/task_performance.png",
        "results/tau_effect.png",
    ]
    for f in figure_files:
        p = Path(f)
        if p.exists():
            p.unlink()
            removed.append(f)
            print(f"  Removed: {f}")
    
    # 3. Remove temporary/test scripts
    print("\n[3] Removing temporary/test scripts...")
    temp_scripts = [
        "scripts/check_plot_data.py",
        "scripts/enhance_research_paper.py",
        "scripts/test_large_experiments.py",
        "scripts/verify_fixed_figures.py",
        "scripts/verify_phase7.py",
    ]
    for f in temp_scripts:
        p = Path(f)
        if p.exists():
            p.unlink()
            removed.append(f)
            print(f"  Removed: {f}")
    
    # 4. Remove Windows artifact files (skip nul as it's a reserved name)
    print("\n[4] Removing Windows artifact files...")
    # Skip 'nul' as it's a Windows reserved name and can't be deleted
    print("  Skipped: nul (Windows reserved name)")
    
    # 5. Remove duplicate table files (keep only in results/tables/)
    print("\n[5] Checking for duplicate tables...")
    # Tables should only be in results/tables/, not results/analysis/tables/
    analysis_tables = Path("results/analysis/tables")
    if analysis_tables.exists():
        # Check if they're duplicates
        main_tables = set(Path("results/tables").glob("*.md"))
        analysis_tables_set = set(analysis_tables.glob("*.md"))
        if main_tables == analysis_tables_set:
            # They're duplicates, remove analysis/tables
            shutil.rmtree(analysis_tables)
            removed.append("results/analysis/tables/")
            print(f"  Removed duplicate: results/analysis/tables/")
    
    # 6. Remove duplicate figures in results/analysis/figures/ (keep only in results/figures/)
    print("\n[6] Checking for duplicate figures...")
    analysis_figures = Path("results/analysis/figures")
    if analysis_figures.exists():
        main_figures = set(Path("results/figures").glob("*.png"))
        analysis_figures_set = set(analysis_figures.glob("*.png"))
        if len(main_figures) > 0 and len(analysis_figures_set) > 0:
            # Remove analysis figures if main figures exist
            shutil.rmtree(analysis_figures)
            removed.append("results/analysis/figures/")
            print(f"  Removed duplicate: results/analysis/figures/")
    
    # Summary
    print("\n" + "=" * 70)
    print("Cleanup Summary")
    print("=" * 70)
    print(f"Total files/directories removed: {len(removed)}")
    print("\nRemoved items:")
    for item in removed:
        print(f"  - {item}")
    
    return removed

if __name__ == "__main__":
    cleanup()
