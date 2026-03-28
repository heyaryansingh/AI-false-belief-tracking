import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def analyze_results():
    input_path = Path("results/metrics/expansion_phase7/results.parquet")
    if not input_path.exists():
        print("No results found.")
        return

    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} episodes.")
    
    # 1. Threshold Protocol (Train/Test Split - Stratified by Environment)
    # Split within each environment to ensure both domains in test set
    df_val_list = []
    df_test_list = []
    
    for env in df["env"].unique():
        env_df = df[df["env"] == env]
        n = len(env_df)
        split_idx = int(n * 0.5)
        df_val_list.append(env_df.iloc[:split_idx])
        df_test_list.append(env_df.iloc[split_idx:])
    
    df_val = pd.concat(df_val_list) if df_val_list else pd.DataFrame()
    df_test = pd.concat(df_test_list) if df_test_list else pd.DataFrame()
    
    print(f"Val N={len(df_val)}, Test N={len(df_test)}")
    
    # 2. Compute Thresholds per Model on Validation Set (Target TPR = 0.8 on Blind Spot)
    thresholds = {}
    for env in ["gridhouse", "virtualhome"]:
        thresholds[env] = {}
        for model in df["model"].unique():
            # Get positive examples (Blind Spot) from Val
            pos_scores = df_val[(df_val["env"] == env) & 
                                (df_val["model"] == model) & 
                                (df_val["condition"] == "blind_spot")]["max_detection_score"]
            
            if len(pos_scores) == 0:
                thresholds[env][model] = 0.5 # Default
                continue
                
            # Find threshold for TPR = 0.8 (80th percentile from top? No, value where 80% are above)
            # np.percentile(scores, 20) would give that roughly.
            threshold = np.percentile(pos_scores, 20) 
            thresholds[env][model] = threshold
    
    # 3. Generate Tables (Test Set Only)
    for env in ["gridhouse", "virtualhome"]:
        print(f"\n=== Domain: {env.upper()} (Test Set N={len(df_test[df_test['env']==env])}) ===")
        print(f"{'Model':<20} | {'AUROC':<6} | {'AUPRC':<6} | {'FPR(Ctrl)':<10} | {'Eff(%)':<6} | {'NetBen':<6} | {'TTD':<6}")
        print("-" * 80)
        
        env_df = df_test[df_test["env"] == env]
        
        for model in ["reactive", "belief_pf", "goal_proxy", "active", "communication", "oracle"]:
            model_df = env_df[env_df["model"] == model]
            if len(model_df) == 0: continue
            
            # AUROC / AUPRC
            y_true = (model_df["condition"] == "blind_spot").astype(int)
            y_score = model_df["max_detection_score"]
            try:
                auroc = roc_auc_score(y_true, y_score)
                auprc = average_precision_score(y_true, y_score)
            except:
                auroc, auprc = 0.5, 0.0
                
            # FPR at specific threshold (on Control condition)
            control_df = model_df[model_df["condition"] == "control"]
            thresh = thresholds[env][model]
            if len(control_df) > 0:
                false_positives = (control_df["max_detection_score"] > thresh).sum()
                fpr = false_positives / len(control_df)
            else:
                fpr = np.nan
                
            # Efficiency (Mean on Blind Spot)
            eff_bs = model_df[model_df["condition"] == "blind_spot"]["efficiency"].mean()
            
            # Net Benefit (Mean on Blind Spot)
            # Net Belief = Saved Wasted Actions - Cost
            # We defined net_benefit in run_expansion.py
            nb_bs = model_df[model_df["condition"] == "blind_spot"]["net_benefit"].mean()
            
            # TTD
            ttd = model_df[model_df["condition"] == "blind_spot"]["time_to_detection"].median()
            
            print(f"{model:<20} | {auroc:.3f}  | {auprc:.3f}  | {fpr:.3f}      | {eff_bs:.3f}  | {nb_bs:.2f}   | {ttd:.1f}")

    # 4. Failure Taxonomy Table
    # "False Belief but Efficient" (Lucky)
    # "Inefficient but No False Belief" (Exploration/Fatigue - proxied by Drift condition where human guesses right?)
    # We can approximate this by looking at Goal Proxy scores vs Ground Truth.
    print("\n=== Failure Taxonomy (Goal Proxy in Blind Spot) ===")
    bs_df = df_test[(df_test["condition"] == "blind_spot") & (df_test["model"] == "goal_proxy")]
    thresh = thresholds["gridhouse"]["goal_proxy"] # Use GH threshold or average?
    
    # FP: Predicted True (Inefficient) but actually... wait, Blind Spot IS True False Belief.
    # So "Failure" here means:
    # FN: Human had False Belief, but was Efficient (Lucky guess) -> Detector said "Safe"
    # We define Efficient as efficiency > 0.9?
    
    lucky = bs_df[(bs_df["efficiency"] > 0.8) & (bs_df["max_detection_score"] < thresh)]
    print(f"Lucky Misses (Efficient but FB): {len(lucky)} / {len(bs_df)}")
    
    # TP: Human had FB, was Inefficient -> Detected
    caught = bs_df[(bs_df["efficiency"] <= 0.8) & (bs_df["max_detection_score"] >= thresh)]
    print(f"Correct Detections (Inefficient & FB): {len(caught)} / {len(bs_df)}")
    
    # Metric of Lucky Ratio
    print(f"Luck Ratio: {len(lucky)/len(bs_df):.2f}")
    
    # 5. Cross-Domain Agreement Analysis
    print("\n=== Cross-Domain Agreement ===")
    models = ["reactive", "belief_pf", "goal_proxy", "active", "communication", "oracle"]
    
    gh_aurocs = []
    vh_aurocs = []
    
    for model in models:
        gh_df = df_test[(df_test["env"] == "gridhouse") & (df_test["model"] == model)]
        vh_df = df_test[(df_test["env"] == "virtualhome") & (df_test["model"] == model)]
        
        if len(gh_df) > 0:
            y_true = (gh_df["condition"] == "blind_spot").astype(int)
            y_score = gh_df["max_detection_score"]
            try:
                gh_aurocs.append(roc_auc_score(y_true, y_score))
            except:
                gh_aurocs.append(0.5)
        else:
            gh_aurocs.append(np.nan)
            
        if len(vh_df) > 0:
            y_true = (vh_df["condition"] == "blind_spot").astype(int)
            y_score = vh_df["max_detection_score"]
            try:
                vh_aurocs.append(roc_auc_score(y_true, y_score))
            except:
                vh_aurocs.append(0.5)
        else:
            vh_aurocs.append(np.nan)
    
    # Compute rank correlation
    from scipy.stats import spearmanr
    valid_mask = ~(np.isnan(gh_aurocs) | np.isnan(vh_aurocs))
    if sum(valid_mask) >= 3:
        rho, p_val = spearmanr(np.array(gh_aurocs)[valid_mask], np.array(vh_aurocs)[valid_mask])
        print(f"Spearman Rank Correlation (GH vs VH): rho={rho:.3f}, p={p_val:.3f}")
        if rho > 0.7:
            print("  -> Strong agreement: Model rankings are consistent across domains.")
        elif rho > 0.4:
            print("  -> Moderate agreement.")
        else:
            print("  -> Weak agreement: Domain-specific effects may exist.")
    else:
        print("  Insufficient data for cross-domain correlation.")
    
    # Save summary CSV
    summary = []
    for i, model in enumerate(models):
        summary.append({"model": model, "GH_AUROC": gh_aurocs[i], "VH_AUROC": vh_aurocs[i]})
    pd.DataFrame(summary).to_csv("results/metrics/expansion_phase7/domain_comparison.csv", index=False)
    print("\nSaved domain comparison to results/metrics/expansion_phase7/domain_comparison.csv")

if __name__ == "__main__":
    analyze_results()
