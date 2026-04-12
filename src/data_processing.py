import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the dataset: dropping nulls, renaming, and re-mapping columns."""
    df = df.copy()
    df.dropna(inplace=True)
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Drop irrelevant columns
    columns_to_drop = ['Timestamp', 'What is your course?', 'Your current year of Study']
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    df.drop(existing_columns, axis=1, inplace=True, errors='ignore')
    
    # Clean CGPA column: Extract the mean of the range instead of lower bound
    def parse_cgpa(x):
        try:
            parts = str(x).split('-')
            if len(parts) == 2:
                return (float(parts[0].strip()) + float(parts[1].strip())) / 2
            return float(parts[0].strip())
        except:
            return 0.0
            
    df['What is your CGPA?'] = df['What is your CGPA?'].apply(parse_cgpa)
    
    # Map binary categorical columns to 0 and 1
    mapping_dict = {'Yes': 1, 'No': 0, 'Female': 1, 'Male': 0}
    
    if 'Choose your gender' in df.columns:
        df['Choose your gender'] = df['Choose your gender'].map(mapping_dict)
    
    cols_to_map = [
        'Do you have Depression?',
        'Do you have Anxiety?',
        'Do you have Panic attack?',
        'Marital status',
        'Did you seek any specialist for a treatment?'
    ]
    
    for col in cols_to_map:
        if col in df.columns:
            df[col] = df[col].map(mapping_dict)
            
    # Drop rows where any of the critical mapped features ended up as NaN
    df.dropna(inplace=True)
    
    return df

def get_features_and_target(df: pd.DataFrame, target_col='Do you have Depression?'):
    """Separates the target from the features."""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y
