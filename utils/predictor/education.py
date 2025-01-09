import pandas as pd

def preprocess_education(df):
    """
    Processes the 'Education' column:
    - Maps education levels to readable categories.
    - Imputes missing/unknown values based on related predictors.
    """
    if 'Education' in df.columns:
        print("[Info] Processing 'Education' column...")

        # Map Education levels
        education_map = {
            "Less than high school": "0_Less than high school",
            "High school graduate": "1_High school graduate",
            "Some college or technical school": "2_Some college",
            "College graduate": "3_College graduate"
        }

        # Debug: Print unique values in original column
        print("[Debug] Original unique values in 'Education':", df['Education'].unique())

        # Apply the mapping
        df['Education'] = df['Education'].map(education_map).fillna("Unknown")

        # Debug: Print frequency of mapped values
        print("[Debug] Mapped 'Education' value counts:")
        print(df['Education'].value_counts())

        # Check for missing mapped categories
        expected_categories = set(education_map.values())
        present_categories = set(df['Education'].unique())
        missing_categories = expected_categories - present_categories

        if missing_categories:
            print(f"[Warning] Missing mapped categories in 'Education': {missing_categories}")

        # Impute 'Unknown' values based on related predictors
        unknown_mask = df['Education'] == "Unknown"
        if unknown_mask.any():
            print("[Info] Imputing 'Unknown' values for 'Education' based on related predictors...")

            # Group-based mode imputation
            df.loc[unknown_mask, 'Education'] = df.groupby(['Gender', 'Race/Ethnicity', 'Income'])['Education'].transform(
                lambda group: group.mode()[0] if not group.mode().empty else "Unknown"
            )

            # Debug: Print updated frequency of 'Unknown'
            print(f"[Debug] Updated 'Unknown' frequency: {df['Education'].value_counts().get('Unknown', 0)}")

        # Ensure 'Education' is treated as categorical
        if not pd.api.types.is_categorical_dtype(df['Education']):
            df['Education'] = pd.Categorical(df['Education'], categories=education_map.values(), ordered=True)
            print("[Debug] Converted 'Education' to categorical type.")

        # Debug: Print final unique values
        print("[Debug] Final unique values in 'Education':", df['Education'].cat.categories)
    else:
        print("[Warning] 'Education' column not found. No changes applied.")

    return df



