import pandas as pd

def impute_race_ethnicity(df):
    """
    Handles missing values in the 'Race/Ethnicity' column, one-hot encodes it, and ensures numeric data types.
    """
    if 'Race/Ethnicity' in df.columns:
        df['Race/Ethnicity'] = df['Race/Ethnicity'].fillna('Unknown')
        race_dummies = pd.get_dummies(df['Race/Ethnicity'], prefix='Race', drop_first=True)
        df = pd.concat([df, race_dummies], axis=1)

        # Ensure all race columns are numeric
        race_cols = [col for col in race_dummies.columns]
        for col in race_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"[Debug] Race/Ethnicity encoded and converted to numeric. New columns: {race_cols}")
    else:
        print("[Warning] 'Race/Ethnicity' column not found.")
    return df