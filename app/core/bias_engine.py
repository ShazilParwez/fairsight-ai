import pandas as pd
from typing import Dict, Any

def analyze_bias(df: pd.DataFrame, target_col: str, sensitive_col: str) -> Dict[str, Any]:
    """
    Computes bias metrics for the given dataframe based on the target and sensitive columns.
    
    Returns a dictionary containing:
    - group_means: Dictionary mapping group names to their mean target value.
    - bias_score: Difference between the max and min group means.
    - disparate_impact: Ratio of the min group mean to the max group mean.
    - is_biased: Boolean indicating if bias_score > 0.1 or disparate_impact < 0.8.
    - risk_level: String categorizing the bias risk (Low / No Bias, Moderate Bias, High Risk Bias).
    """
    group_means_series = df.groupby(sensitive_col)[target_col].mean()
    group_means = group_means_series.to_dict()
    
    max_mean = group_means_series.max()
    min_mean = group_means_series.min()
    
    bias_score = max_mean - min_mean
    
    # Calculate Disparate Impact (min_mean / max_mean)
    # Handle division by zero if max_mean is 0
    if max_mean == 0:
        disparate_impact = 1.0 if min_mean == 0 else 0.0
    else:
        disparate_impact = min_mean / max_mean
        
    is_biased = (bias_score > 0.1) or (disparate_impact < 0.8)
    
    # Determine risk level based on bias score
    if bias_score < 0.05:
        risk_level = "Low / No Bias"
    elif bias_score <= 0.15:
        risk_level = "Moderate Bias"
    else:
        risk_level = "High Risk Bias"
    
    return {
        "group_means": group_means,
        "bias_score": bias_score,
        "disparate_impact": disparate_impact,
        "is_biased": is_biased,
        "risk_level": risk_level
    }
