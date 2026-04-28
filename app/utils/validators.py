import pandas as pd

def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str]:
    """Validates the overall dataframe."""
    if df is None or df.empty:
        return False, "The dataset is empty or not loaded properly."
    return True, ""

def validate_target_column(df: pd.DataFrame, target_col: str) -> tuple[bool, str]:
    """Ensures target column is numeric and handles missing values."""
    if target_col not in df.columns:
        return False, f"Target column '{target_col}' not found in dataset."
    
    # Check for missing values in the target column
    if df[target_col].isnull().all():
        return False, f"Target column '{target_col}' contains only missing values."
        
    if df[target_col].isnull().any():
        return False, f"Target column '{target_col}' contains missing values. Please clean the data."

    # Check if the column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return False, f"Target column '{target_col}' must be numeric."

    return True, ""

def validate_sensitive_column(df: pd.DataFrame, sensitive_col: str) -> tuple[bool, str]:
    """Validates the sensitive column, including sufficient group diversity."""
    if sensitive_col not in df.columns:
        return False, f"Sensitive column '{sensitive_col}' not found in dataset."
        
    if df[sensitive_col].isnull().any():
        return False, f"Sensitive column '{sensitive_col}' contains missing values. Please clean the data."

    # Check for group diversity (at least 2 groups)
    unique_groups = df[sensitive_col].nunique()
    if unique_groups < 2:
        return False, f"Sensitive column '{sensitive_col}' must have at least two distinct groups for comparison."

    return True, ""
