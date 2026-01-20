"""Statistical utilities for robust analysis and reporting.

This module provides functions for:
- Bootstrap confidence intervals for AUROC and other metrics
- Effect size calculations (Cohen's d)
- Statistical significance tests (paired t-test, Wilcoxon)
- Generic confidence interval computation

# Fix: Bootstrap CI added to stabilize AUROC variance (Phase 10)
"""

from typing import Tuple, Optional, List, Union, Dict, Any
import numpy as np
from scipy import stats


def compute_auroc_with_ci(
    y_true: Union[List, np.ndarray],
    y_pred: Union[List, np.ndarray],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Compute AUROC with bootstrap confidence interval.
    
    # Fix: Bootstrap CI added to stabilize AUROC variance - uses resampling
    # to compute robust confidence bounds instead of simple mean ± SD.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted scores (probabilities)
        n_bootstrap: Number of bootstrap samples (default 10000)
        confidence_level: Confidence level for CI (default 0.95)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with keys:
        - 'auroc': Mean AUROC across bootstrap samples
        - 'ci_lower': Lower bound of confidence interval
        - 'ci_upper': Upper bound of confidence interval
        - 'std': Standard deviation of bootstrap samples
        - 'n_samples': Number of valid bootstrap samples
    """
    from sklearn.metrics import roc_auc_score
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'auroc': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n_samples': 0,
        }
    
    # Check for single class
    if len(np.unique(y_true)) < 2:
        return {
            'auroc': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n_samples': 0,
        }
    
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    aurocs = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = rng.choice(n, n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]
        
        # Skip if single class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            auroc = roc_auc_score(y_true_boot, y_pred_boot)
            aurocs.append(auroc)
        except ValueError:
            continue
    
    if len(aurocs) == 0:
        return {
            'auroc': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n_samples': 0,
        }
    
    aurocs = np.array(aurocs)
    mean_auroc = np.mean(aurocs)
    std_auroc = np.std(aurocs, ddof=1)
    
    # Compute CI using percentile method (more robust than t-distribution for bootstrap)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(aurocs, 100 * alpha / 2)
    ci_upper = np.percentile(aurocs, 100 * (1 - alpha / 2))
    
    return {
        'auroc': mean_auroc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': std_auroc,
        'n_samples': len(aurocs),
    }


def compute_bootstrap_ci(
    values: Union[List, np.ndarray],
    statistic: str = 'mean',
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a given statistic.
    
    # Fix: Generic bootstrap CI function for any metric aggregation.
    
    Args:
        values: Array of values to compute CI for
        statistic: Statistic to compute ('mean', 'median')
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for CI
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with statistic value and CI bounds
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]  # Remove NaN values
    
    if len(values) == 0:
        return {
            'value': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'n': 0,
        }
    
    rng = np.random.RandomState(random_state)
    n = len(values)
    
    stat_func = np.mean if statistic == 'mean' else np.median
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        bootstrap_stats.append(stat_func(values[idx]))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return {
        'value': stat_func(values),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': np.std(bootstrap_stats, ddof=1),
        'n': n,
    }


def compute_confidence_interval(
    values: Union[List, np.ndarray],
    confidence_level: float = 0.95,
) -> Dict[str, float]:
    """Compute confidence interval using t-distribution.
    
    # Fix: T-distribution CI for small samples (n < 30), normal for larger.
    
    Args:
        values: Array of values
        confidence_level: Confidence level (default 0.95)
        
    Returns:
        Dictionary with mean, ci_lower, ci_upper, std, sem, n
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return {
            'mean': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'std': np.nan,
            'sem': np.nan,
            'n': 0,
        }
    
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0
    
    if n > 1:
        # Use t-distribution for CI
        t_crit = stats.t.ppf((1 + confidence_level) / 2, df=n - 1)
        margin = t_crit * sem
        ci_lower = mean - margin
        ci_upper = mean + margin
    else:
        ci_lower = mean
        ci_upper = mean
    
    return {
        'mean': mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': std,
        'sem': sem,
        'n': n,
    }


def effect_size(
    a: Union[List, np.ndarray],
    b: Union[List, np.ndarray],
    pooled: bool = True,
) -> float:
    """Compute Cohen's d effect size between two groups.
    
    # Fix: Effect size (Cohen's d) for pairwise model comparisons.
    # Interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small, 
    # 0.5-0.8 = medium, > 0.8 = large
    
    Args:
        a: First group values
        b: Second group values
        pooled: Whether to use pooled standard deviation (default True)
        
    Returns:
        Cohen's d effect size (positive if mean(a) > mean(b))
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    
    if len(a) == 0 or len(b) == 0:
        return np.nan
    
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    if pooled:
        # Pooled standard deviation (assumes equal variance)
        var_a = np.var(a, ddof=1) if len(a) > 1 else 0
        var_b = np.var(b, ddof=1) if len(b) > 1 else 0
        n_a = len(a)
        n_b = len(b)
        
        # Pooled variance
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = np.sqrt(pooled_var)
    else:
        # Simple average of standard deviations
        std_a = np.std(a, ddof=1) if len(a) > 1 else 0
        std_b = np.std(b, ddof=1) if len(b) > 1 else 0
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    
    if pooled_std == 0:
        return 0.0 if mean_a == mean_b else np.inf * np.sign(mean_a - mean_b)
    
    return (mean_a - mean_b) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
        
    Returns:
        String interpretation
    """
    abs_d = abs(d)
    if np.isnan(d):
        return "undefined"
    elif abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def paired_ttest(
    a: Union[List, np.ndarray],
    b: Union[List, np.ndarray],
) -> Dict[str, float]:
    """Perform paired t-test between two groups.
    
    # Fix: Paired t-test for statistical significance testing between models.
    
    Args:
        a: First group values (must be same length as b)
        b: Second group values
        
    Returns:
        Dictionary with statistic, p_value, significant (at 0.05)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Remove pairs with NaN in either
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    
    if len(a) < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n': len(a),
        }
    
    statistic, p_value = stats.ttest_rel(a, b)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n': len(a),
    }


def wilcoxon_test(
    a: Union[List, np.ndarray],
    b: Union[List, np.ndarray],
) -> Dict[str, float]:
    """Perform Wilcoxon signed-rank test between two groups.
    
    # Fix: Non-parametric alternative to paired t-test for non-normal data.
    
    Args:
        a: First group values (must be same length as b)
        b: Second group values
        
    Returns:
        Dictionary with statistic, p_value, significant (at 0.05)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Remove pairs with NaN in either
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    
    # Wilcoxon requires at least some non-zero differences
    diff = a - b
    if len(diff) < 2 or np.all(diff == 0):
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n': len(a),
        }
    
    try:
        statistic, p_value = stats.wilcoxon(a, b)
    except ValueError:
        # Can happen if all differences are zero
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n': len(a),
        }
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n': len(a),
    }


def independent_ttest(
    a: Union[List, np.ndarray],
    b: Union[List, np.ndarray],
    equal_var: bool = False,
) -> Dict[str, float]:
    """Perform independent samples t-test (Welch's t-test by default).
    
    Args:
        a: First group values
        b: Second group values  
        equal_var: Assume equal variance (default False for Welch's test)
        
    Returns:
        Dictionary with statistic, p_value, significant (at 0.05)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    
    if len(a) < 2 or len(b) < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n_a': len(a),
            'n_b': len(b),
        }
    
    statistic, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_a': len(a),
        'n_b': len(b),
    }


def mann_whitney_test(
    a: Union[List, np.ndarray],
    b: Union[List, np.ndarray],
) -> Dict[str, float]:
    """Perform Mann-Whitney U test (non-parametric independent samples test).
    
    Args:
        a: First group values
        b: Second group values
        
    Returns:
        Dictionary with statistic, p_value, significant (at 0.05)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    
    if len(a) < 1 or len(b) < 1:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n_a': len(a),
            'n_b': len(b),
        }
    
    statistic, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_a': len(a),
        'n_b': len(b),
    }


def format_ci(
    mean: float,
    ci_lower: float,
    ci_upper: float,
    precision: int = 3,
) -> str:
    """Format mean with confidence interval as string.
    
    Args:
        mean: Mean value
        ci_lower: CI lower bound
        ci_upper: CI upper bound
        precision: Decimal precision
        
    Returns:
        Formatted string like "0.718 [0.650, 0.770]"
    """
    if np.isnan(mean):
        return "N/A"
    return f"{mean:.{precision}f} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"


def format_p_value(p: float, threshold: float = 0.001) -> str:
    """Format p-value with significance indicators.
    
    Args:
        p: P-value
        threshold: Threshold below which to show "< threshold"
        
    Returns:
        Formatted string like "0.023*" or "< 0.001***"
    """
    if np.isnan(p):
        return "N/A"
    
    if p < threshold:
        stars = "***"
        return f"< {threshold}{stars}"
    elif p < 0.001:
        stars = "***"
    elif p < 0.01:
        stars = "**"
    elif p < 0.05:
        stars = "*"
    else:
        stars = ""
    
    return f"{p:.3f}{stars}"


def summary_statistics(
    values: Union[List, np.ndarray],
    name: str = "metric",
) -> Dict[str, Any]:
    """Compute comprehensive summary statistics for a metric.
    
    # Fix: Comprehensive statistics for publication-ready reporting.
    
    Args:
        values: Array of values
        name: Name of the metric
        
    Returns:
        Dictionary with all summary statistics
    """
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return {
            'name': name,
            'n': 0,
            'mean': np.nan,
            'std': np.nan,
            'sem': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'q25': np.nan,
            'q75': np.nan,
            'iqr': np.nan,
        }
    
    ci_result = compute_confidence_interval(values)
    
    return {
        'name': name,
        'n': len(values),
        'mean': np.mean(values),
        'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
        'sem': ci_result['sem'],
        'ci_lower': ci_result['ci_lower'],
        'ci_upper': ci_result['ci_upper'],
        'median': np.median(values),
        'min': np.min(values),
        'max': np.max(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
        'iqr': np.percentile(values, 75) - np.percentile(values, 25),
    }
