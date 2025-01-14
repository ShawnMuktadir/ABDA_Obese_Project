# Preprocess the LocationDesc column
def preprocess_location(df):
    """
    Processes the 'LocationDesc' column:
    - Converts to a categorical predictor and encodes it.
    """
    if 'LocationDesc' in df.columns:
        print("[Info] Processing 'LocationDesc' column...")

        # Convert to categorical type
        df['LocationDesc'] = df['LocationDesc'].astype('category')

        # Encode categories as integers
        df['LocationDesc_Encoded'] = df['LocationDesc'].cat.codes

        print("[Debug] Final unique values in 'LocationDesc':", df['LocationDesc'].cat.categories)
        print("[Debug] Encoded LocationDesc values:")
        print(df[['LocationDesc', 'LocationDesc_Encoded']].drop_duplicates())
    else:
        print("[Warning] 'LocationDesc' column not found. No changes applied.")
    return df