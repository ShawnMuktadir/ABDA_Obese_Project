def impute_gender(df):
    """
    Encodes the 'Gender' column as binary and imputes missing values.
    """
    if 'Gender' in df.columns:
        print("[Start] Handling 'Gender' column.")
        gender_map = {'male': 1, 'female': 0}
        df['Gender'] = df['Gender'].str.lower().map(gender_map)
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    else:
        print("[Warning] 'Gender' column not found.")
    return df