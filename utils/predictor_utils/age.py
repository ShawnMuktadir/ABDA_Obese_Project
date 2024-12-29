import pandas as pd
import numpy as np

# === Helper Functions ===

def normalize_column(column, column_name):
    """
    Normalizes a numeric column to have mean=0 and standard deviation=1.
    Returns the normalized column, its mean, and standard deviation for reverse-scaling.
    """
    mean = column.mean(skipna=True)
    std = column.std(skipna=True)
    print(f"[Debug] Normalizing column: {column_name} - Mean: {mean:.2f}, Std: {std:.2f}")

    if std > 0:
        normalized_column = (column - mean) / std
    else:
        print(f"[Warning] Standard deviation of column '{column_name}' is 0. Skipping normalization.")
        normalized_column = column  # Do not normalize if std=0

    print(f"[Debug] First 5 original values in '{column_name}': {column.head(5).values}")
    print(f"[Debug] First 5 normalized values in '{column_name}': {normalized_column.head(5).values}")

    return normalized_column, mean, std


def convert_age_to_midpoint(age_value):
    """
    Converts age ranges like '25 - 34' into midpoints,
    handles '65+' cases, or returns NaN for missing/invalid values.
    """
    if pd.isna(age_value):
        return np.nan  # Missing age
    age_value = str(age_value).strip()

    if "-" in age_value:  # Range like '25 - 34'
        try:
            low, high = map(int, age_value.replace(" ", "").split("-"))
            return (low + high) / 2
        except ValueError:
            print(f"[Debug] Unable to parse range: {age_value}")
            return np.nan
    elif "65" in age_value:  # Handle '65+' case
        return 70  # Approximation
    elif age_value.isdigit():
        return float(age_value)
    else:
        print(f"[Debug] Unexpected format for age: {age_value}")
        return np.nan  # Unexpected format


def process_age_column(df):
    """
    Cleans and imputes the 'Age(years)' column using group-based imputation for higher accuracy.
    """
    if 'Age(years)' in df.columns:
        print("[Debug] Raw 'Age(years)' values before processing:")
        print(df['Age(years)'].head(20))

        df['Age(years)'] = df['Age(years)'].apply(convert_age_to_midpoint)
        print("[Debug] Processed 'Age(years)' values after conversion:")
        print(df['Age(years)'].unique())

        missing_before = df['Age(years)'].isna().mean() * 100
        print(f"[Debug] Missing percentage in 'Age(years)' before imputation: {missing_before:.2f}%")

        if missing_before > 50:
            print("[Warning] High percentage of missing 'Age(years)'. Using group-based mean imputation.")
            group_means = df.groupby(['Gender', 'Race/Ethnicity'])['Age(years)'].transform('mean')
            df['Age(years)'] = df['Age(years)'].fillna(group_means)

        age_mean = df['Age(years)'].mean(skipna=True)
        df['Age(years)'] = df['Age(years)'].fillna(age_mean)

        missing_after = df['Age(years)'].isna().mean() * 100
        print(f"[Debug] Missing percentage in 'Age(years)' after imputation: {missing_after:.2f}%")

    return df