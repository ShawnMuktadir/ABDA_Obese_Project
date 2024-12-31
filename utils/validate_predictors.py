import pandas as pd

def validate_predictors(df, predictors):
    """
    Validates that all predictors are numeric.
    """
    for col in predictors:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"[Error] Predictor '{col}' is not numeric. Column values:\n{df[col].head()}")
