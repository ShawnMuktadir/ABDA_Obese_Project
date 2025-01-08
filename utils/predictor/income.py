import pandas as pd
import numpy as np

# === Helper Functions ===

def preprocess_income(df):
    """
    Processes and imputes the 'Income' column.
    """
    if 'Income' in df.columns:
        print("[Start] Handling 'Income' column.")

        def convert_income_to_midpoint(income_value):
            if pd.isna(income_value):
                return np.nan
            income_value = str(income_value).strip()

            if "-" in income_value:
                try:
                    low, high = income_value.replace("$", "").replace(",", "").split("-")
                    return (float(low) + float(high)) / 2
                except ValueError:
                    print(f"[Error] Failed to parse range: {income_value}")
                    return np.nan

            if "greater" in income_value:
                return 75000

            return np.nan

        df['Income'] = df['Income'].apply(convert_income_to_midpoint)

        missing_before = df['Income'].isna().mean() * 100
        print(f"[Debug] Missing percentage in 'Income' before imputation: {missing_before:.2f}%")

        income_mode = df.groupby(['Gender', 'Race/Ethnicity'])['Income'].transform(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        df['Income'] = df['Income'].fillna(income_mode)

        missing_after = df['Income'].isna().mean() * 100
        print(f"[Debug] Missing percentage in 'Income' after imputation: {missing_after:.2f}%")
    else:
        print("[Warning] 'Income' column not found.")
    return df