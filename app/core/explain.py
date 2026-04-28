from typing import Dict, Any

def explain_bias(bias_results: Dict[str, Any], sensitive_col: str, target_col: str) -> Dict[str, str]:
    """
    Generates a structured human-readable explanation based on the bias detection results.
    Returns a dictionary with 'summary', 'metrics', 'actions', and 'why_it_matters'.
    """
    is_biased = bias_results["is_biased"]
    group_means = bias_results["group_means"]
    bias_score = bias_results["bias_score"]
    disparate_impact = bias_results["disparate_impact"]
    
    # Sort groups by mean value
    sorted_groups = sorted(group_means.items(), key=lambda item: item[1])
    lowest_group, min_mean = sorted_groups[0]
    highest_group, max_mean = sorted_groups[-1]
    
    # Calculate percentage difference safely
    if max_mean > 0:
        pct_diff = ((max_mean - min_mean) / max_mean) * 100
    else:
        pct_diff = 0.0

    # Strong impact statement
    impact_statement = "Even small biases at scale can lead to large systemic inequalities."
    
    # Semi-dynamic Why This Matters
    target_lower = target_col.lower()
    if any(keyword in target_lower for keyword in ['hire', 'offer', 'employ']):
        domain_impact = "hiring decisions, potentially excluding qualified candidates based on demographics."
    elif any(keyword in target_lower for keyword in ['loan', 'credit', 'approve', 'mortgage']):
        domain_impact = "financial approvals, potentially denying credit access unfairly."
    elif any(keyword in target_lower for keyword in ['admit', 'school', 'educat', 'enroll']):
        domain_impact = "educational opportunities, potentially limiting academic access unfairly."
    elif any(keyword in target_lower for keyword in ['arrest', 'recidivism', 'crime']):
        domain_impact = "criminal justice outcomes, which can have devastating life consequences."
    else:
        domain_impact = "real-world scenarios, potentially leading to unfair treatment or resource allocation."

    why_it_matters = f"This level of bias can lead to unfair outcomes in {domain_impact} {impact_statement}"
    
    if is_biased:
        summary = (
            f"<b>Bias Detected:</b> The model favors <b>{highest_group}</b> over <b>{lowest_group}</b> "
            f"by approximately <b>{pct_diff:.1f}%</b> across {sensitive_col} groups."
        )
        metrics = (
            f"<ul>"
            f"<li><b>Most Advantaged Group:</b> '{highest_group}' (mean {target_col}: {max_mean:.3f})</li>"
            f"<li><b>Most Disadvantaged Group:</b> '{lowest_group}' (mean {target_col}: {min_mean:.3f})</li>"
            f"<li><b>Bias Score:</b> {bias_score:.3f} (Difference in outcomes)</li>"
            f"<li><b>Disparate Impact:</b> {disparate_impact:.3f} (Values below 0.8 typically indicate unfairness)</li>"
            f"</ul>"
        )
        actions = (
            "<ul>"
            f"<li>Investigate the data collection process for the '{lowest_group}' group.</li>"
            f"<li>Consider collecting more balanced training samples for '{lowest_group}'.</li>"
            "<li>Apply bias mitigation techniques (e.g., reweighing) before training.</li>"
            "<li>Consult with domain experts to understand potential historical biases in the data.</li>"
            "</ul>"
        )
    else:
        summary = f"<b>No Significant Bias Detected:</b> Outcomes for {target_col} appear relatively fair across {sensitive_col} groups."
        metrics = (
            f"<ul>"
            f"<li><b>Highest Group:</b> '{highest_group}' (mean: {max_mean:.3f})</li>"
            f"<li><b>Lowest Group:</b> '{lowest_group}' (mean: {min_mean:.3f})</li>"
            f"<li><b>Bias Score:</b> {bias_score:.3f} (Below threshold of 0.1)</li>"
            f"<li><b>Disparate Impact:</b> {disparate_impact:.3f} (Above threshold of 0.8)</li>"
            f"</ul>"
        )
        actions = (
            "<ul>"
            "<li>Continue monitoring the model's performance on new data.</li>"
            "<li>Regularly re-evaluate as the underlying population distributions may shift.</li>"
            "</ul>"
        )
        
    return {
        "summary": summary,
        "metrics": metrics,
        "actions": actions,
        "why_it_matters": why_it_matters
    }
