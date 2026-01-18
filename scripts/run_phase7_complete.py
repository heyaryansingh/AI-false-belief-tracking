"""Complete Phase 7 execution: Generate data, run experiments, create visualizations.

This script is designed to be resumable - it checks what's already done and continues.
Run with: python scripts/run_phase7_complete.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

# Progress file for resumability
PROGRESS_FILE = Path(__file__).parent.parent / "results" / "phase7_progress.json"


def load_progress():
    """Load progress from checkpoint file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "episodes_generated": 0,
        "experiments_completed": [],
        "analysis_done": False,
        "visualizations_done": False,
        "tables_done": False,
    }


def save_progress(progress):
    """Save progress to checkpoint file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def generate_episodes(progress, num_episodes=1000):
    """Generate episodes for all experimental conditions."""
    from bsa.envs.gridhouse.env import GridHouseEnvironment
    from bsa.envs.gridhouse.episode_generator import GridHouseEpisodeGenerator
    from bsa.envs.gridhouse.recorder import EpisodeRecorder

    print("\n" + "="*60)
    print("STEP 1: Episode Generation")
    print("="*60)

    output_dir = Path(__file__).parent.parent / "data" / "episodes" / "phase7_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check existing episodes
    existing = list(output_dir.glob("*.parquet"))
    if len(existing) >= num_episodes:
        print(f"  Already have {len(existing)} episodes, skipping generation")
        progress["episodes_generated"] = len(existing)
        return progress

    # Generate remaining episodes
    start_from = len(existing)
    print(f"  Starting from episode {start_from}, generating to {num_episodes}")

    env = GridHouseEnvironment(seed=42)
    recorder = EpisodeRecorder()

    # Distribution of conditions
    conditions = [
        {"intervention_type": None, "name": "control"},
        {"intervention_type": "relocate", "name": "false_belief"},
        {"intervention_type": "relocate", "name": "seen_relocation"},  # We'll simulate seen
    ]

    episodes_per_condition = num_episodes // 3

    for cond_idx, condition in enumerate(conditions):
        cond_start = cond_idx * episodes_per_condition
        cond_end = cond_start + episodes_per_condition

        # Skip if already done
        if start_from >= cond_end:
            continue

        actual_start = max(start_from, cond_start)
        print(f"\n  Generating {condition['name']} episodes ({actual_start} to {cond_end})")

        generator = GridHouseEpisodeGenerator(
            env=env,
            seed=42 + cond_idx * 1000,
            tau_range=(5, 20),
            drift_probability=1.0 if condition["intervention_type"] else 0.0,
        )

        for i in range(actual_start - cond_start, episodes_per_condition):
            episode = generator.generate_episode(intervention_type=condition["intervention_type"])

            # Add condition to metadata
            episode.metadata["condition"] = condition["name"]

            # Save
            episode_num = cond_start + i
            output_path = output_dir / f"episode_{episode_num:05d}_{condition['name']}.parquet"
            recorder.save_episode(episode, output_path)

            if (i + 1) % 50 == 0:
                print(f"    Generated {i+1}/{episodes_per_condition} {condition['name']} episodes")

    # Count final
    final_count = len(list(output_dir.glob("*.parquet")))
    print(f"\n  Total episodes generated: {final_count}")
    progress["episodes_generated"] = final_count
    save_progress(progress)
    return progress


def run_experiments(progress):
    """Run experiments with all helper agents."""
    from bsa.envs.gridhouse.recorder import EpisodeRecorder
    from bsa.agents.helper.reactive import ReactiveHelper
    from bsa.agents.helper.goal_only import GoalOnlyHelper
    from bsa.agents.helper.belief_sensitive import BeliefSensitiveHelper
    from bsa.experiments.evaluator import EpisodeEvaluator

    print("\n" + "="*60)
    print("STEP 2: Running Experiments")
    print("="*60)

    episode_dir = Path(__file__).parent.parent / "data" / "episodes" / "phase7_fixed"
    results_dir = Path(__file__).parent.parent / "results" / "metrics" / "phase7_fixed"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load episodes
    episode_files = sorted(episode_dir.glob("*.parquet"))
    print(f"  Found {len(episode_files)} episodes")

    recorder = EpisodeRecorder()
    evaluator = EpisodeEvaluator()

    # Define models
    models = {
        "reactive": lambda seed: ReactiveHelper(seed=seed),
        "goal_only": lambda seed: GoalOnlyHelper(seed=seed),
        "belief_pf": lambda seed: BeliefSensitiveHelper(seed=seed),
    }

    # Check what's already done
    results_file = results_dir / "results.parquet"
    if results_file.exists():
        existing_df = pd.read_parquet(results_file)
        completed_combos = set(zip(existing_df["model"], existing_df["episode_id"]))
        print(f"  Found {len(completed_combos)} existing results")
    else:
        existing_df = None
        completed_combos = set()

    all_results = []

    for ep_idx, ep_file in enumerate(episode_files):
        episode = recorder.load_episode(ep_file)
        condition = episode.metadata.get("condition", "unknown")

        for model_name, model_factory in models.items():
            combo = (model_name, episode.episode_id)
            if combo in completed_combos:
                continue

            helper = model_factory(seed=ep_idx)
            helper.reset()

            metrics = evaluator.evaluate_episode(episode, helper)

            result = {
                "experiment_name": "phase7_fixed",
                "model": model_name,
                "condition": condition,
                "episode_id": episode.episode_id,
                "goal_id": episode.goal_id,
                "tau": episode.tau,
                "intervention_type": episode.intervention_type,
                "false_belief_created": episode.metadata.get("false_belief_created", False),
                "false_belief_steps": episode.metadata.get("false_belief_steps", 0),
                **metrics,
            }
            all_results.append(result)

        if (ep_idx + 1) % 100 == 0:
            print(f"    Processed {ep_idx+1}/{len(episode_files)} episodes")

            # Save intermediate results
            if all_results:
                new_df = pd.DataFrame(all_results)
                if existing_df is not None:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    combined_df = new_df
                combined_df.to_parquet(results_file, index=False)
                existing_df = combined_df
                all_results = []
                print(f"      Saved checkpoint ({len(combined_df)} total results)")

    # Final save
    if all_results:
        new_df = pd.DataFrame(all_results)
        if existing_df is not None:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_parquet(results_file, index=False)
        print(f"\n  Final results saved: {len(combined_df)} total")

    progress["experiments_completed"] = list(models.keys())
    save_progress(progress)
    return progress


def create_visualizations(progress):
    """Create publication-quality visualizations."""
    print("\n" + "="*60)
    print("STEP 3: Creating Visualizations")
    print("="*60)

    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # Set publication-quality defaults
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    sns.set_style("whitegrid")

    results_file = Path(__file__).parent.parent / "results" / "metrics" / "phase7_fixed" / "results.parquet"
    figures_dir = Path(__file__).parent.parent / "results" / "figures" / "phase7_fixed"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not results_file.exists():
        print("  ERROR: Results file not found")
        return progress

    df = pd.read_parquet(results_file)
    print(f"  Loaded {len(df)} results")

    # Filter to false_belief condition for main analysis
    df_fb = df[df["condition"] == "false_belief"].copy()

    # Color palette
    model_colors = {"reactive": "#E74C3C", "goal_only": "#3498DB", "belief_pf": "#27AE60"}
    model_order = ["reactive", "goal_only", "belief_pf"]
    model_labels = {"reactive": "Reactive", "goal_only": "Goal-Only", "belief_pf": "Belief-Sensitive"}

    # 1. AUROC Comparison Bar Chart
    print("  Creating AUROC comparison chart...")
    fig, ax = plt.subplots(figsize=(8, 5))

    auroc_data = df_fb.groupby("model")["false_belief_detection_auroc"].agg(["mean", "std", "count"]).reset_index()
    auroc_data["se"] = auroc_data["std"] / np.sqrt(auroc_data["count"])

    x_pos = np.arange(len(model_order))
    bars = ax.bar(x_pos,
                  [auroc_data[auroc_data["model"] == m]["mean"].values[0] for m in model_order],
                  yerr=[auroc_data[auroc_data["model"] == m]["se"].values[0] * 1.96 for m in model_order],
                  color=[model_colors[m] for m in model_order],
                  capsize=5, edgecolor="black", linewidth=1)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in model_order])
    ax.set_ylabel("False Belief Detection AUROC")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random Chance")
    ax.legend()
    ax.set_title("False Belief Detection Performance")

    # Add value labels
    for bar, m in zip(bars, model_order):
        height = bar.get_height()
        val = auroc_data[auroc_data["model"] == m]["mean"].values[0]
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.03, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / "auroc_comparison.png")
    plt.savefig(figures_dir / "auroc_comparison.pdf")
    plt.close()

    # 2. Precision-Recall Comparison
    print("  Creating precision-recall chart...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes, ["intervention_precision", "intervention_recall"],
                                 ["Intervention Precision", "Intervention Recall"]):
        data = df_fb.groupby("model")[metric].agg(["mean", "std", "count"]).reset_index()
        data["se"] = data["std"] / np.sqrt(data["count"])

        bars = ax.bar(x_pos,
                      [data[data["model"] == m]["mean"].values[0] for m in model_order],
                      yerr=[data[data["model"] == m]["se"].values[0] * 1.96 for m in model_order],
                      color=[model_colors[m] for m in model_order],
                      capsize=5, edgecolor="black", linewidth=1)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([model_labels[m] for m in model_order])
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)

        for bar, m in zip(bars, model_order):
            height = bar.get_height()
            val = data[data["model"] == m]["mean"].values[0]
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.03, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall.png")
    plt.savefig(figures_dir / "precision_recall.pdf")
    plt.close()

    # 3. AUROC Distribution Violin Plot
    print("  Creating AUROC distribution plot...")
    fig, ax = plt.subplots(figsize=(8, 5))

    plot_data = df_fb[df_fb["false_belief_detection_auroc"].notna()].copy()
    plot_data["model_label"] = plot_data["model"].map(model_labels)

    sns.violinplot(data=plot_data, x="model", y="false_belief_detection_auroc",
                   order=model_order, palette=model_colors, ax=ax)
    ax.set_xticklabels([model_labels[m] for m in model_order])
    ax.set_ylabel("AUROC Distribution")
    ax.set_xlabel("")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Distribution of AUROC Scores Across Episodes")

    plt.tight_layout()
    plt.savefig(figures_dir / "auroc_violin.png")
    plt.savefig(figures_dir / "auroc_violin.pdf")
    plt.close()

    # 4. Heatmap of Performance by Condition
    print("  Creating condition heatmap...")
    fig, ax = plt.subplots(figsize=(10, 6))

    pivot_data = df.groupby(["model", "condition"])["false_belief_detection_auroc"].mean().unstack()
    pivot_data = pivot_data.reindex(model_order)
    pivot_data.index = [model_labels[m] for m in model_order]

    sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.4, vmax=1.0, ax=ax, cbar_kws={"label": "AUROC"})
    ax.set_title("AUROC by Model and Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(figures_dir / "condition_heatmap.png")
    plt.savefig(figures_dir / "condition_heatmap.pdf")
    plt.close()

    # 5. Comprehensive Summary Figure (2x2)
    print("  Creating summary figure...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 5a. AUROC bars
    ax = axes[0, 0]
    auroc_means = [auroc_data[auroc_data["model"] == m]["mean"].values[0] for m in model_order]
    auroc_ses = [auroc_data[auroc_data["model"] == m]["se"].values[0] * 1.96 for m in model_order]
    ax.bar(x_pos, auroc_means, yerr=auroc_ses, color=[model_colors[m] for m in model_order],
           capsize=5, edgecolor="black", linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in model_order])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("A) False Belief Detection AUROC")

    # 5b. Precision
    ax = axes[0, 1]
    prec_data = df_fb.groupby("model")["intervention_precision"].agg(["mean", "std", "count"]).reset_index()
    prec_data["se"] = prec_data["std"] / np.sqrt(prec_data["count"])
    prec_means = [prec_data[prec_data["model"] == m]["mean"].values[0] for m in model_order]
    prec_ses = [prec_data[prec_data["model"] == m]["se"].values[0] * 1.96 for m in model_order]
    ax.bar(x_pos, prec_means, yerr=prec_ses, color=[model_colors[m] for m in model_order],
           capsize=5, edgecolor="black", linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in model_order])
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.1)
    ax.set_title("B) Intervention Precision")

    # 5c. FPR
    ax = axes[1, 0]
    fpr_data = df_fb.groupby("model")["false_belief_detection_fpr"].agg(["mean", "std", "count"]).reset_index()
    fpr_data["se"] = fpr_data["std"] / np.sqrt(fpr_data["count"])
    fpr_means = [fpr_data[fpr_data["model"] == m]["mean"].values[0] for m in model_order]
    fpr_ses = [fpr_data[fpr_data["model"] == m]["se"].values[0] * 1.96 for m in model_order]
    ax.bar(x_pos, fpr_means, yerr=fpr_ses, color=[model_colors[m] for m in model_order],
           capsize=5, edgecolor="black", linewidth=1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in model_order])
    ax.set_ylabel("False Positive Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("C) False Positive Rate (lower is better)")

    # 5d. AUROC violin
    ax = axes[1, 1]
    sns.violinplot(data=plot_data, x="model", y="false_belief_detection_auroc",
                   order=model_order, palette=model_colors, ax=ax)
    ax.set_xticklabels([model_labels[m] for m in model_order])
    ax.set_ylabel("AUROC")
    ax.set_xlabel("")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("D) AUROC Distribution")

    plt.tight_layout()
    plt.savefig(figures_dir / "summary_figure.png")
    plt.savefig(figures_dir / "summary_figure.pdf")
    plt.close()

    # 6. Statistical comparison
    print("  Performing statistical tests...")
    stats_results = []

    for metric in ["false_belief_detection_auroc", "intervention_precision", "intervention_recall"]:
        belief_pf_vals = df_fb[df_fb["model"] == "belief_pf"][metric].dropna()
        for baseline in ["reactive", "goal_only"]:
            baseline_vals = df_fb[df_fb["model"] == baseline][metric].dropna()
            if len(belief_pf_vals) > 0 and len(baseline_vals) > 0:
                t_stat, p_val = stats.ttest_ind(belief_pf_vals, baseline_vals)
                effect_size = (belief_pf_vals.mean() - baseline_vals.mean()) / np.sqrt(
                    (belief_pf_vals.std()**2 + baseline_vals.std()**2) / 2
                )
                stats_results.append({
                    "metric": metric,
                    "comparison": f"belief_pf vs {baseline}",
                    "t_statistic": t_stat,
                    "p_value": p_val,
                    "effect_size": effect_size,
                    "belief_pf_mean": belief_pf_vals.mean(),
                    "baseline_mean": baseline_vals.mean(),
                })

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(figures_dir / "statistical_tests.csv", index=False)
    print(f"  Statistical tests saved to {figures_dir / 'statistical_tests.csv'}")

    print(f"\n  All figures saved to {figures_dir}")
    progress["visualizations_done"] = True
    save_progress(progress)
    return progress


def generate_tables(progress):
    """Generate publication-quality tables in Markdown and LaTeX."""
    print("\n" + "="*60)
    print("STEP 4: Generating Tables")
    print("="*60)

    results_file = Path(__file__).parent.parent / "results" / "metrics" / "phase7_fixed" / "results.parquet"
    tables_dir = Path(__file__).parent.parent / "results" / "tables" / "phase7_fixed"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if not results_file.exists():
        print("  ERROR: Results file not found")
        return progress

    df = pd.read_parquet(results_file)

    # Main results table
    print("  Generating main results table...")
    metrics = ["false_belief_detection_auroc", "intervention_precision", "intervention_recall",
               "false_belief_detection_fpr"]
    metric_labels = {
        "false_belief_detection_auroc": "AUROC",
        "intervention_precision": "Precision",
        "intervention_recall": "Recall",
        "false_belief_detection_fpr": "FPR"
    }
    model_labels = {"reactive": "Reactive", "goal_only": "Goal-Only", "belief_pf": "Belief-Sensitive"}

    # Filter to false_belief condition
    df_fb = df[df["condition"] == "false_belief"]

    # Create summary table
    rows = []
    for model in ["reactive", "goal_only", "belief_pf"]:
        model_data = df_fb[df_fb["model"] == model]
        row = {"Model": model_labels[model]}
        for metric in metrics:
            vals = model_data[metric].dropna()
            if len(vals) > 0:
                mean = vals.mean()
                std = vals.std()
                row[metric_labels[metric]] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[metric_labels[metric]] = "N/A"
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # Save as markdown
    md_content = "# Main Results: False Belief Detection Performance\n\n"
    md_content += results_df.to_markdown(index=False)
    md_content += "\n\n*Results on false_belief condition. Values shown as mean ± std.*\n"

    with open(tables_dir / "main_results.md", "w") as f:
        f.write(md_content)

    # Save as LaTeX
    latex_content = r"""\begin{table}[h]
\centering
\caption{False Belief Detection Performance}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
Model & AUROC & Precision & Recall & FPR \\
\midrule
"""
    for _, row in results_df.iterrows():
        latex_content += f"{row['Model']} & {row['AUROC']} & {row['Precision']} & {row['Recall']} & {row['FPR']} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(tables_dir / "main_results.tex", "w") as f:
        f.write(latex_content)

    # Condition comparison table
    print("  Generating condition comparison table...")
    cond_rows = []
    for condition in ["control", "false_belief", "seen_relocation"]:
        for model in ["reactive", "goal_only", "belief_pf"]:
            cond_data = df[(df["condition"] == condition) & (df["model"] == model)]
            auroc_vals = cond_data["false_belief_detection_auroc"].dropna()
            if len(auroc_vals) > 0:
                cond_rows.append({
                    "Condition": condition.replace("_", " ").title(),
                    "Model": model_labels[model],
                    "AUROC": f"{auroc_vals.mean():.3f}",
                    "N": len(auroc_vals),
                })

    cond_df = pd.DataFrame(cond_rows)
    pivot_df = cond_df.pivot(index="Model", columns="Condition", values="AUROC")

    md_content = "# Performance by Experimental Condition\n\n"
    md_content += pivot_df.to_markdown()
    md_content += "\n\n*AUROC values by model and condition.*\n"

    with open(tables_dir / "condition_comparison.md", "w") as f:
        f.write(md_content)

    # Statistical significance table
    print("  Generating statistical significance table...")
    stats_file = Path(__file__).parent.parent / "results" / "figures" / "phase7_fixed" / "statistical_tests.csv"
    if stats_file.exists():
        stats_df = pd.read_csv(stats_file)

        md_content = "# Statistical Significance Tests\n\n"
        md_content += "Comparing Belief-Sensitive model against baselines.\n\n"
        md_content += "| Metric | Comparison | t-statistic | p-value | Effect Size (Cohen's d) |\n"
        md_content += "|--------|------------|-------------|---------|-------------------------|\n"

        for _, row in stats_df.iterrows():
            sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else ""
            md_content += f"| {row['metric'].replace('_', ' ').title()} | {row['comparison']} | {row['t_statistic']:.3f} | {row['p_value']:.4f}{sig} | {row['effect_size']:.3f} |\n"

        md_content += "\n*Significance: * p<0.05, ** p<0.01, *** p<0.001*\n"

        with open(tables_dir / "statistical_significance.md", "w") as f:
            f.write(md_content)

    print(f"\n  Tables saved to {tables_dir}")
    progress["tables_done"] = True
    save_progress(progress)
    return progress


def main():
    """Run complete Phase 7 pipeline."""
    print("\n" + "#"*60)
    print("# PHASE 7 COMPLETE EXECUTION")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)

    progress = load_progress()
    print(f"\nLoaded progress: {progress}")

    start_time = time.time()

    try:
        # Step 1: Generate episodes
        progress = generate_episodes(progress, num_episodes=1000)

        # Step 2: Run experiments
        progress = run_experiments(progress)

        # Step 3: Create visualizations
        progress = create_visualizations(progress)

        # Step 4: Generate tables
        progress = generate_tables(progress)

        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("PHASE 7 COMPLETE!")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print("="*60)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved. Run again to continue.")
        save_progress(progress)
        return 1

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        save_progress(progress)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
