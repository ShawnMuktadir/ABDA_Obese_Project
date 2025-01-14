import pandas as pd

def preprocess_gender(df):
    """
    Processes the 'Gender' column:
    - Maps gender values to readable categories.
    - Imputes missing/unknown values based on related predictors.
    - Handles cases with entirely missing 'Gender' data.
    """
    if 'Gender' in df.columns:
        print("[Info] Processing 'Gender' column...")

        # Map Gender values
        gender_map = {
            "male": "Male",
            "female": "Female",
            "unknown": "Unknown"
        }

        # Debug: Print unique values in original column
        print("[Debug] Original unique values in 'Gender':", df['Gender'].unique())

        # Apply the mapping
        df['Gender'] = df['Gender'].str.lower().map(gender_map).fillna("Unknown")

        # Debug: Print frequency of mapped values
        print("[Debug] Mapped 'Gender' value counts:")
        print(df['Gender'].value_counts())

        # Impute 'Unknown' values based on related predictors
        unknown_mask = df['Gender'] == "Unknown"
        if unknown_mask.any():
            print("[Info] Imputing 'Unknown' values for 'Gender' based on related predictors...")

            # Group-based mode imputation
            df.loc[unknown_mask, 'Gender'] = df.groupby(['Race/Ethnicity', 'Income'])['Gender'].transform(
                lambda group: group.mode()[0] if not group.mode().empty else "Unknown"
            )

            # Debug: Print updated frequency of 'Unknown'
            print(f"[Debug] Updated 'Unknown' frequency: {df['Gender'].value_counts().get('Unknown', 0)}")

        # Ensure 'Gender' is treated as categorical
        if not pd.api.types.is_categorical_dtype(df['Gender']):
            df['Gender'] = pd.Categorical(df['Gender'], categories=gender_map.values(), ordered=True)
            print("[Debug] Converted 'Gender' to categorical type.")

        # Handle cases where 'Gender' remains empty after processing
        if df['Gender'].isnull().all():
            print("[Warning] 'Gender' column is entirely empty; filling with 'Unknown'.")
            df['Gender'] = "Unknown"
    else:
        print("[Warning] 'Gender' column not found. No changes applied.")

    return df
