def preprocess_location(df):
    """
    Processes the 'LocationDesc' column:
    - Converts to a categorical predictor.
    """
    if 'LocationDesc' in df.columns:
        print("[Info] Processing 'LocationDesc' column...")
        df['LocationDesc'] = df['LocationDesc'].astype('category')
        print("[Debug] Final unique values in 'LocationDesc':", df['LocationDesc'].cat.categories)
    else:
        print("[Warning] 'LocationDesc' column not found. No changes applied.")
    return df
