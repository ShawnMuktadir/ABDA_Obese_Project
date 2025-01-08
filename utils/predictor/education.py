import pandas as pd

def preprocess_education(df):
    """
    Handles the Education column:
    - Maps education levels to readable categories.
    - Imputes missing/unknown values based on related predictors.
    - Prints value counts for education categories.
    """
    if 'Education' in df.columns:
        print("[Start] Processing 'Education' column.")

        # Map Education levels
        education_map = {
            "Less than high school": "0_Less than high school",
            "High school graduate": "1_High school graduate",
            "Some college": "2_Some college",
            "College graduate": "3_College graduate"
        }
        df['Education'] = df['Education'].map(education_map)

        # Mark unknowns and count their frequency
        df['Education'] = df['Education'].fillna("Unknown")
        print("[Debug] Unique values in 'Education':", df['Education'].unique())

        # Debug: Print value counts in desired format
        value_counts = df['Education'].value_counts()
        print("[Debug] Updated value counts for 'Education':")
        for category, count in value_counts.items():
            print(f"{category:<25} {count:>10}")

        # Impute 'Unknown' based on other predictors
        unknown_mask = df['Education'] == "Unknown"
        if unknown_mask.any():
            print("[Info] Imputing missing 'Education' values based on related predictors...")

            def impute_education(group):
                # Replace "Unknown" with mode if available, otherwise keep "Unknown"
                mode_value = group.mode()
                if not mode_value.empty:
                    return group.replace("Unknown", mode_value[0])
                return group

            try:
                df.loc[unknown_mask, 'Education'] = df.groupby(['Gender', 'Race/Ethnicity', 'Income'])['Education'].apply(impute_education)
            except ValueError as e:
                print(f"[Error] Imputation failed due to: {e}. Using 'Unknown' as fallback.")
                df['Education'] = df['Education'].fillna("Unknown")

            # Debug: Print updated value counts after imputation
            print("[Debug] Updated value counts for 'Education' after imputation:")
            value_counts_after = df['Education'].value_counts()
            for category, count in value_counts_after.items():
                print(f"{category:<25} {count:>10}")

        # Convert to categorical type for modeling
        education_categories = [
            "0_Less than high school", "1_High school graduate",
            "2_Some college", "3_College graduate", "Unknown"
        ]
        df['Education'] = pd.Categorical(df['Education'], categories=education_categories, ordered=True)
        print("[Debug] 'Education' column converted to ordered categorical.")

    else:
        print("[Warning] 'Education' column not found in the dataset. No changes applied.")

    return df
