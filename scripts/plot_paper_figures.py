import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_paper_figures():
    # Load data
    data_path = Path("results/metrics/phase11_realism/results.parquet")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return

    df = pd.read_parquet(data_path)
    output_dir = Path("results/figures_phase12_scientific")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Global Style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    colors = {"reactive": "#95a5a6", "goal_only": "#3498db", "belief_pf": "#e74c3c"}
    
    # =========================================================================
    # Figure 1: Detection Performance (The "Blind Spot" Visualized)
    # Panel A: AUROC (Sensitivity) - Only for False Belief conditions
    # Panel B: FPR (Specificity) - For ALL conditions
    # =========================================================================
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Filter for AUROC plot (Control has undefined AUROC, so remove it)
    auroc_df = df[df["condition"] != "control"]
    sns.barplot(
        data=auroc_df, x="condition", y="auroc", hue="model", 
        palette=colors, errorbar=("ci", 95), capsize=.1, ax=axes[0]
    )
    axes[0].set_title("A. Detection Sensitivity (AUROC)\n(Higher is Better)", fontweight='bold')
    axes[0].set_ylim(0.4, 0.9)
    axes[0].axhline(0.5, ls='--', color='black', label='Random Chance')
    axes[0].set_ylabel("AUROC")
    axes[0].legend(title="Agent Model")
    
    # Filter for FPR plot (All conditions)
    # We want to show Control has roughly 0.0 FPR (Perfect Specificity)
    sns.barplot(
        data=df, x="condition", y="fpr", hue="model", 
        palette=colors, errorbar=("ci", 95), capsize=.1, ax=axes[1]
    )
    axes[1].set_title("B. False Alarm Rate (FPR)\n(Lower is Better)", fontweight='bold')
    axes[1].set_ylim(0.0, 0.2)
    axes[1].set_ylabel("False Positive Rate")
    axes[1].get_legend().remove() # Unified legend in panel A
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_detection_panel.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # Figure 2: Efficiency Distribution (Bimodality Analysis)
    # =========================================================================
    plt.figure(figsize=(12, 7))
    sns.violinplot(
        data=df, x="model", y="efficiency", hue="condition",
        palette="viridis", split=False, inner="quartile"
    )
    plt.title("Task Efficiency Distribution by Model and Condition", fontweight='bold')
    plt.ylim(0.4, 1.0)
    plt.ylabel("Efficiency (Optimal / Actual Steps)")
    plt.savefig(output_dir / "fig2_efficiency_violin.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # Figure 3: Time to Detection Dynamics
    # Comparison of TTD for Reactive vs Belief-Sensitive
    # =========================================================================
    plt.figure(figsize=(10, 6))
    ttd_df = df[df["time_to_detection"].notna()]
    if not ttd_df.empty:
        sns.boxplot(
            data=ttd_df, x="model", y="time_to_detection",
            palette=colors, showfliers=False
        )
        plt.title("Time-to-Detection Latency", fontweight='bold')
        plt.ylabel("Timesteps after False Belief Onset")
        plt.xlabel("Agent Model")
        plt.savefig(output_dir / "fig3_ttd_boxplot.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # Figure 4: Intervention Outcome Stacked Bar
    # =========================================================================
    # Calculate rates of intervention types per condition
    # This requires some aggregation of the raw data columns
    # For now, let's plot "Number of Interventions"
    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=df, x="condition", y="interventions", hue="model",
        palette=colors, errorbar=("ci", 95), capsize=.1
    )
    plt.title("Intervention Frequency by Condition", fontweight='bold')
    plt.ylabel("Avg Interventions per Episode")
    plt.savefig(output_dir / "fig4_intervention_freq.png", dpi=300)
    plt.close()

    print(f"Figures saved to {output_dir}")

if __name__ == "__main__":
    plot_paper_figures()
