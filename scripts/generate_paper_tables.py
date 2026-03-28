import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

def ci_str(data):
    """Returns mean [95% CI] string."""
    if len(data) < 2:
        return f"{data.mean():.3f} [-]"
    res = stats.bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=1000)
    low, high = res.confidence_interval.low, res.confidence_interval.high
    return f"{data.mean():.3f} [{low:.3f}, {high:.3f}]"

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

def generate_tables():
    data_path = Path("results/metrics/phase11_realism/results.parquet")
    if not data_path.exists():
        print("Data not found.")
        return
    
    df = pd.read_parquet(data_path)
    # Filter out NaNs for AUROC (Control condition)
    df_auroc = df.dropna(subset=['auroc'])
    
    output_dir = Path("paper/tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # TABLE 1: Detection Performance (AUROC)
    # Goal-Only vs Belief-Sensitive
    models = ["goal_only", "belief_pf", "reactive"]
    table1 = []
    headers = ["Model", "AUROC (Mean [95% CI])", "N (Episodes)", "vs Goal-Only (d)"]
    
    goal_auroc = df_auroc[df_auroc['model'] == 'goal_only']['auroc']
    
    for m in models:
        dat = df_auroc[df_auroc['model'] == m]['auroc']
        d_val = cohen_d(dat, goal_auroc) if m != 'goal_only' else 0.0
        table1.append([
            m.replace("_", " ").title(),
            ci_str(dat),
            len(dat),
            f"{d_val:.3f}"
        ])
        
    df_t1 = pd.DataFrame(table1, columns=headers)
    with open(output_dir / "table1_detection.md", "w") as f:
        f.write("# Table 1: Detection Performance\n\n")
        f.write(df_t1.to_markdown(index=False))
        
    # TABLE 2: Efficiency (All Conditions)
    table2 = []
    headers2 = ["Model", "Condition", "Efficiency [95% CI]", "Wasted Actions"]
    
    for cond in df['condition'].unique():
        for m in models:
            sub = df[(df['model'] == m) & (df['condition'] == cond)]
            eff = sub['efficiency']
            # Approximation for wasted actions: (1/eff) - 1 * avg_steps (~50)
            # Not exact, but descriptive
            wasted = (1.0 / eff.mean() - 1.0) * 50 if eff.mean() > 0 else 0
            
            table2.append([
                m, cond, ci_str(eff), f"{wasted:.1f}"
            ])
            
    df_t2 = pd.DataFrame(table2, columns=headers2)
    with open(output_dir / "table2_efficiency.md", "w") as f:
        f.write("# Table 2: Task Efficiency\n\n")
        f.write(df_t2.to_markdown(index=False))

    # TABLE 3: False Positive Rate (Control Condition)
    headers3 = ["Model", "FPR (Mean [95% CI])", "N"]
    table3 = []
    control_df = df[df['condition'] == 'control']
    
    for m in models:
        dat = control_df[control_df['model'] == m]['fpr']
        table3.append([m, ci_str(dat), len(dat)])
        
    df_t3 = pd.DataFrame(table3, columns=headers3)
    with open(output_dir / "table3_specificity.md", "w") as f:
        f.write("# Table 3: Specificity (Control Condition FPR)\n\n")
        f.write(df_t3.to_markdown(index=False))

    print(f"Tables generated in {output_dir}")

if __name__ == "__main__":
    generate_tables()
