import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np

def plot_expansion():
    output_dir = Path("results/figures_expansion")
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    
    path = Path("results/metrics/expansion_phase7/results.parquet")
    if not path.exists(): return
    df = pd.read_parquet(path)
    
    # 1. Multi-Panel ROC (GridHouse vs VirtualHome)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, env in enumerate(["gridhouse", "virtualhome"]):
        ax = axes[idx]
        sub_df = df[df["env"] == env]
        
        models = sub_df["model"].unique()
        for model in models:
            m_df = sub_df[sub_df["model"] == model]
            y_true = (m_df["condition"] == "blind_spot").astype(int)
            y_score = m_df["max_detection_score"]
            if len(set(y_true)) < 2: continue
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{model} ({roc_auc:.2f})")
            
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_title(f"ROC: {env.title()}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_roc_comparison.png")
    plt.close()
    
    # 2. Cost-Benefit Analysis (Net Benefit)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="model", y="net_benefit", hue="env")
    plt.title("Net Benefit (Saved Wasted Actions - Cost)")
    plt.xticks(rotation=45)
    plt.savefig(output_dir / "fig5_cost_benefit.png")
    plt.close()
    
    print("Expansion figures generated.")

if __name__ == "__main__":
    plot_expansion()
