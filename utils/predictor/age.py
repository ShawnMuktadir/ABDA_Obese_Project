import pandas as pd
import numpy as np

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

        # Convert age ranges to numeric midpoints
        df['Age(years)'] = df['Age(years)'].apply(convert_age_to_midpoint)
        print("[Debug] Processed 'Age(years)' values after conversion to midpoints:")
        print(df['Age(years)'].head(20))

        # Check missing percentages
        missing_before = df['Age(years)'].isna().mean() * 100
        print(f"[Debug] Missing percentage in 'Age(years)' before imputation: {missing_before:.2f}%")

        if missing_before > 50:
            print("[Warning] High percentage of missing 'Age(years)'. Using group-based mean imputation.")
            group_means = df.groupby(['Gender', 'Race/Ethnicity'])['Age(years)'].transform('mean')
            df['Age(years)'] = df['Age(years)'].fillna(group_means)

        # Global mean imputation for any remaining missing values
        age_mean = df['Age(years)'].mean(skipna=True)
        df['Age(years)'] = df['Age(years)'].fillna(age_mean)

        missing_after = df['Age(years)'].isna().mean() * 100
        print(f"[Debug] Missing percentage in 'Age(years)' after imputation: {missing_after:.2f}%")

        # Confirm that the column is numeric
        if pd.api.types.is_numeric_dtype(df['Age(years)']):
            print("[Info] 'Age(years)' column successfully converted to numeric.")
        else:
            print("[Error] 'Age(years)' column conversion to numeric failed.")
    else:
        print("[Warning] 'Age(years)' column not found in the DataFrame. Adding placeholder values.")
        df['Age(years)'] = -1  # Placeholder for missing column
        print("[Debug] Added 'Age(years)' column with default value -1.")

    return df
