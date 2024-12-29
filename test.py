import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('dir/Nutrition_Physical_Activity_and_Obesity.csv')


# 1. Convert Age Ranges to Midpoints
def convert_age_to_midpoint(age_value):
    """
    Converts age ranges like '25 - 34' into midpoints,
    handles numeric values, or returns NaN for missing/invalid values.
    """
    if pd.isna(age_value):  # If value is NaN, return NaN
        return np.nan
    elif "-" in str(age_value):  # Convert ranges like "25 - 34" to midpoints
        try:
            low, high = map(int, age_value.split(" - "))
            return (low + high) / 2
        except ValueError:
            return np.nan
    else:  # Return numeric or convert valid strings to numeric
        try:
            return float(age_value)
        except ValueError:
            return np.nan


# Apply midpoint conversion to the Age(years) column
df['Age(years)'] = df['Age(years)'].apply(convert_age_to_midpoint)

# 2. Calculate Group-Based Imputation Values
# Group by Gender, Income, and Race/Ethnicity to compute the mean 'Age(years)'
group_means = df.groupby(['Gender', 'Income', 'Race/Ethnicity'])['Age(years)'].mean()
print("Group-based mean ages:\n", group_means)


# 3. Fill Missing Values Using Group Averages
def fill_missing_age(row):
    """
    Fills missing Age(years) using group-level averages
    if available, otherwise falls back to global mean.
    """
    if not pd.isna(row['Age(years)']):  # If age is not missing, keep the value
        return row['Age(years)']

    # Fetch group mean for the specific row's Gender, Income, and Race/Ethnicity
    group_mean = group_means.get((row['Gender'], row['Income'], row['Race/Ethnicity']), np.nan)

    # Use the group mean if it exists, else fallback to global mean
    if not pd.isna(group_mean):
        return group_mean
    else:
        return df['Age(years)'].mean()  # Fallback to global mean


# Apply the imputation function to fill missing Age(years)
df['Age(years)'] = df.apply(fill_missing_age, axis=1)

# 4. Verify and Output Results
# Verify if missing values are filled
missing_count = df['Age(years)'].isna().sum()
print(f"Missing values left in 'Age(years)': {missing_count}")

# Optional: Check dataset statistics for Age(years)
print("Age(years) Column Statistics:\n", df['Age(years)'].describe())

# Save the processed DataFrame to a new CSV file if needed
# Remove `#` to enable saving
# df.to_csv('processed_trimmed_file.csv', index=False)


# Analyze all unique values in the 'Question' column to identify relevant inputs for obesity
unique_questions = df['Question'].unique()

# Display all unique questions for further analysis
unique_questions_list = unique_questions.tolist()
unique_questions_list
