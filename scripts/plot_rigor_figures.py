import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_rigor_figures():
    output_dir = Path("results/figures_rigor")
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    
    # 1. Ablation Plot (AUROC vs Particles)
    ablation_path = Path("results/metrics/ablations/results.parquet")
    if ablation_path.exists():
        df_ab = pd.read_parquet(ablation_path)
        plt.figure(figsize=(8, 6))
        
        # Calculate mean and CI
        sns.lineplot(data=df_ab, x="particles", y="auroc", marker="o", errorbar=("ci", 95))
        plt.title("Ablation: Detection Sensitivity vs Compute", fontweight='bold')
        plt.xlabel("Number of Particles")
        plt.ylabel("AUROC")
        plt.ylim(0.4, 0.8)
        plt.axhline(0.5, ls='--', color='gray', label='Random Chance')
        plt.xscale('log')
        plt.xticks([50, 100, 500, 1000], [50, 100, 500, 1000])
        plt.legend()
        plt.savefig(output_dir / "fig_ablation_particles.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated Ablation Plot.")

    # 2. ROC Curves (All Models)
    baseline_path = Path("results/metrics/baselines_roc/results.parquet")
    if baseline_path.exists() and ablation_path.exists():
        df_base = pd.read_parquet(baseline_path)
        
        # Get "Best" Belief model (1000 particles) for comparison
        df_belief = df_ab[df_ab["particles"] == 1000].copy()
        
        # Combine
        df_all = pd.concat([df_base, df_belief])
        
        plt.figure(figsize=(8, 8))
        
        models = {
            "goal_only": ("Goal Proxy (Efficiency)", "#2ecc71"), # Green
            "belief_pf": ("Explicit Belief (Particles=1000)", "#e74c3c"), # Red
            "reactive": ("Reactive Baseline", "#95a5a6") # Grey
        }
        
        for model, (label, color) in models.items():
            sub = df_all[df_all["model"] == model]
            if len(sub) == 0: continue
            
            y_true = (sub["condition"] == "false_belief").astype(int)
            y_score = sub["max_detection_score"]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC)', fontweight='bold')
        plt.legend(loc="lower right")
        plt.savefig(output_dir / "fig_roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated ROC Curves.")
        
    # 3. Efficiency Distributions (Goal-Only Model)
    # Histogram of efficiency for True Belief vs False Belief episodes
    if baseline_path.exists():
        df_goal = df_base[df_base["model"] == "goal_only"]
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            data=df_goal, x="efficiency", hue="condition", 
            fill=True, common_norm=False, palette="viridis", alpha=0.5
        )
        plt.title("Behavioral Signature: Efficiency Distribution", fontweight='bold')
        plt.xlabel("Task Efficiency (Optimal/Actual)")
        plt.axvline(0.7, ls=':', color='black', label='Decision Threshold') # Approximate threshold
        plt.legend(title="Condition")
        plt.savefig(output_dir / "fig_efficiency_dist.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Generated Efficiency Distribution.")

if __name__ == "__main__":
    plot_rigor_figures()
