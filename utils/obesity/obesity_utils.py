import pandas as pd

def process_obesity_with_75th_percentile(df):
    """
    Processes the dataset to create the 'Obese' column using the 75th percentile threshold.
    """
    obesity_df = df[df['Class'] == 'Obesity / Weight Status']
    obesity_data = obesity_df[obesity_df['Question'] == 'Percent of adults aged 18 years and older who have obesity']
    obesity_data.loc[:, 'Data_Value'] = obesity_data['Data_Value'].fillna(obesity_data['Data_Value'].mean())
    quantile_threshold = obesity_data['Data_Value'].quantile(0.75)
    obesity_data.loc[:, 'Obese'] = (obesity_data['Data_Value'] >= quantile_threshold).astype(int)

    indirect_questions = [
        'Percent of adults who engage in no leisure-time physical activity',
        'Percent of adults who report consuming fruit less than one time daily',
        'Percent of adults who report consuming vegetables less than one time daily'
    ]
    indirect_data = df[df['Question'].isin(indirect_questions)]

    combined_data = pd.merge(
        obesity_data[['YearStart', 'LocationDesc', 'Obese']],
        indirect_data[['YearStart', 'LocationDesc', 'Question', 'Data_Value']],
        on=['YearStart', 'LocationDesc'],
        how='left'
    )

    combined_data = combined_data.pivot_table(
        index=['YearStart', 'LocationDesc', 'Obese'],
        columns='Question',
        values='Data_Value'
    ).reset_index()

    combined_data.rename(columns={
        'Percent of adults who engage in no leisure-time physical activity': 'No_Physical_Activity',
        'Percent of adults who report consuming fruit less than one time daily': 'Low_Fruit_Consumption',
        'Percent of adults who report consuming vegetables less than one time daily': 'Low_Veg_Consumption'
    }, inplace=True)

    # Merge back demographic and race columns
    demographic_cols = ['Age(years)', 'Income', 'Gender', 'Race/Ethnicity']
    combined_data = pd.merge(
        combined_data,
        df[['YearStart', 'LocationDesc'] + demographic_cols].drop_duplicates(),
        on=['YearStart', 'LocationDesc'],
        how='left'
    )

    # Fill missing values: separately handle numeric and non-numeric columns
    for col in combined_data.select_dtypes(include=['number']).columns:
        combined_data[col].fillna(combined_data[col].mean(), inplace=True)

    for col in combined_data.select_dtypes(exclude=['number']).columns:
        combined_data[col].fillna("Unknown", inplace=True)

    return combined_data