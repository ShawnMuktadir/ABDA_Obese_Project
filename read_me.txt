import pandas as pd

# Update the file path to match where you saved the dataset on your machine
file_path = '../../Documents/Md. Muktadir/Semester/Winter Semester 24_25/ABDA/project_24_25/Nutrition_Physical_Activity_and_Obesity.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Trim the dataset to the first 10 rows (or another number of rows)
trimmed_data = data.head(50)

# Save the trimmed dataset to a new file
new_file_path = 'trimmed_file.csv'
trimmed_data.to_csv(new_file_path, index=False)

# Display confirmation
print(f"Trimmed dataset saved to {new_file_path}")


# Display the structure of the dataset
print(data.info())

# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Display the first few rows
print(data.head())

# Handle missing values
data = data.dropna()

# Explore the dataset
print(data.info())
print(data.head())
print(data.isnull().sum())
print("The columns are:\n", data.columns.tolist())


# Set Pandas to display all columns and rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows
pd.set_option('display.expand_frame_repr', False)  # Disable wrapping

# Example: Print the full DataFrame
print(data)
