# === Helper Functions ===


def impute_data_value(df):
    """
    Handles missing values in the 'Data_Value' column.
    """
    if 'Data_Value' in df.columns:
        print("[Imputation] Handling missing 'Data_Value' values.")
        mean_value = df['Data_Value'].mean()
        df['Data_Value'] = df['Data_Value'].fillna(mean_value)
        print(f"[Debug] Remaining missing 'Data_Value' filled with mean: {mean_value:.2f}")
    else:
        print("[Warning] 'Data_Value' column not found.")
    return df