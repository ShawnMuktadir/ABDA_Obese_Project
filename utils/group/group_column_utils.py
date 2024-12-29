import pandas as pd

def ensure_numeric_group_columns(df, group_cols):
    """
    Ensures all specified group columns are converted to numeric codes for hierarchical modeling.
    """
    for col in group_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
            print(f"[Debug] Converted '{col}' to numeric codes.")
        else:
            print(f"[Warning] '{col}' column not found in the dataset!")
    return df
    """
    Converts specified columns into numeric codes for hierarchical modeling.
    """
    for col in group_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
            print(f"[Debug] Converted '{col}' to numeric codes.")
        else:
            print(f"[Warning] '{col}' column not found in the dataset!")
    return df
    """
    Converts the 'LocationDesc' column into numeric codes for hierarchical modeling.
    """
    if 'LocationDesc' in df.columns:
        df['LocationDesc'] = pd.Categorical(df['LocationDesc']).codes
        print("[Debug] Converted 'LocationDesc' to numeric codes.")
    else:
        print("[Warning] 'LocationDesc' column not found in the dataset!")
    return df


def compute_group_proportions(df, group_col, target):
    """
    Computes the proportion of the target (Obese) within each group.
    """
    group_proportions = df.groupby(group_col)[target].mean()
    return group_proportions