def preprocess_education(df):
    """
    Encodes the Education column into numeric format with readable mapping for visualization.
    """
    if 'Education' in df.columns:
        print("[Start] Handling 'Education' column.")

        # Map Education levels to numeric values (Ordinal Encoding)
        education_map = {
            "Less than high school": 0,
            "High school graduate": 1,
            "Some college": 2,
            "College graduate": 3
        }
        df['Education'] = df['Education'].map(education_map)

        # Handle missing values
        df['Education'] = df['Education'].fillna(-1)  # Use -1 for missing education levels

        # Convert to categorical type for easier mapping later
        df['Education'] = df['Education'].astype('category')
        print("[Debug] Unique values in 'Education':", df['Education'].unique())
    else:
        print("[Warning] 'Education' column not found.")

    return df
