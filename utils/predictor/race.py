import pandas as pd

def preprocess_race_ethnicity(df):
    """
    Handles missing values in the 'Race/Ethnicity' column, one-hot encodes it,
    and ensures numeric data types.
    """
    if 'Race/Ethnicity' in df.columns:
        print("[Info] Processing 'Race/Ethnicity' column...")

        # Fill missing values with 'Unknown'
        df['Race/Ethnicity'] = df['Race/Ethnicity'].fillna('Unknown')

        # Convert to categorical type for consistency
        if not isinstance(df['Race/Ethnicity'].dtype, pd.CategoricalDtype):
            df['Race/Ethnicity'] = df['Race/Ethnicity'].astype('category')
            print("[Debug] Converted 'Race/Ethnicity' to categorical type.")

        # Debug: Print unique categories
        print("[Debug] Unique categories in 'Race/Ethnicity':", df['Race/Ethnicity'].cat.categories)

        # Generate one-hot encoded columns
        race_dummies = pd.get_dummies(df['Race/Ethnicity'], prefix='Race', drop_first=False)
        print("[Debug] One-hot encoded columns created:", race_dummies.columns.tolist())

        # Merge one-hot encoded columns into the DataFrame
        df = pd.concat([df, race_dummies], axis=1)

        # Debug: Print all columns in DataFrame after adding Race_ columns
        print("[Debug] Columns in DataFrame after adding Race_ columns:", df.columns.tolist())

        # Ensure the one-hot encoded columns are numeric
        for col in race_dummies.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop the original 'Race/Ethnicity' column if not needed
        # Uncomment the next line if the original column should not be retained
        # df.drop(columns=['Race/Ethnicity'], inplace=True)

        print("[Info] 'Race/Ethnicity' processing completed successfully.")
    else:
        print("[Warning] 'Race/Ethnicity' column not found in the dataset. No changes applied.")

    return df

def debug_race_ethnicity_values(df):
    """
    Prints all unique values in the 'Race/Ethnicity' column after processing.
    """
    if 'Race/Ethnicity' in df.columns:
        print("[Debug] Unique values in 'Race/Ethnicity' column:")
        print(df['Race/Ethnicity'].unique())
    else:
        print("[Error] 'Race/Ethnicity' column not found in the dataset.")
