import pandas as pd


def ensure_numeric_group_columns(df, group_cols):
    """
    Ensures that the specified group columns are converted to numeric codes.
    Arguments:
        df: DataFrame containing the group columns.
        group_cols: List of columns to ensure as numeric.
    Returns:
        df: Modified DataFrame with numeric group columns.
    """
    for col in group_cols:
        if col in df.columns:
            print(f"[Info] Processing column: {col}")

            # Convert to category if not already categorical
            if not isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype('category')

            # Get existing categories and ensure 'Unknown' is included
            existing_categories = df[col].cat.categories
            if 'Unknown' not in existing_categories:
                df[col] = df[col].cat.set_categories(list(existing_categories) + ['Unknown'])

            # Fill missing values with 'Unknown'
            df[col] = df[col].fillna('Unknown')

            # Convert to numeric codes
            df[col] = df[col].cat.codes
            print(f"[Debug] Unique values in '{col}': {df[col].unique()}")
        else:
            print(f"[Warning] Column '{col}' not found in DataFrame.")

    return df

def compute_group_proportions(df, group_col, target):
    """
    Computes the proportion of the target variable within each group.
    Arguments:
        df: DataFrame containing the data.
        group_col: Column to group by (e.g., 'LocationDesc', 'Race/Ethnicity').
        target: Target column (e.g., 'Obese').
    Returns:
        A Series with group-wise proportions.
    """
    # Validate columns
    if group_col not in df.columns:
        raise KeyError(f"[Error] The group column '{group_col}' is not found in the DataFrame!")
    if target not in df.columns:
        raise KeyError(f"[Error] The target column '{target}' is not found in the DataFrame!")

    # Convert group_col to categorical if not already
    if df[group_col].dtype != 'category':
        df[group_col] = df[group_col].astype('category')
        print(f"[Info] Converted '{group_col}' to categorical type.")

    # Ensure target is numeric
    if not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError(f"[Error] The target column '{target}' must be numeric for group mean calculation!")

    # Compute group proportions
    group_proportions = df.groupby(group_col)[target].mean()
    print(f"[Debug] Group proportions calculated for '{group_col}':")
    print(group_proportions)

    return group_proportions

