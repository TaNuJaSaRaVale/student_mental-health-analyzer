import pandas as pd
import numpy as np
import random
import os

file_path = 'Student Mental health.csv'
df = pd.read_csv(file_path)

# Impute NaN age if any
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
df = df.dropna()

gender_col = 'Choose your gender'
females = df[df[gender_col] == 'Female']
males = df[df[gender_col] == 'Male']

target_per_gender = 500

def generate_data(subset, target):
    augmented = []
    current_count = len(subset)
    
    courses = subset['What is your course?'].unique().tolist()
    years = subset['Your current year of Study'].unique().tolist()
    cgpas = subset['What is your CGPA?'].unique().tolist()
    
    symptom_cols = ['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Marital status', 'Did you seek any specialist for a treatment?']
    symptom_combos = subset[symptom_cols].values.tolist()
    
    for i in range(target):
        base_row = subset.iloc[random.randint(0, current_count - 1)].copy()
        
        # Standard normal distribution based jittering for Age
        base_row['Age'] = max(18, min(30, int(np.random.normal(subset['Age'].mean(), subset['Age'].std()))))
        
        # Keep correlation for CGPA, Course, Year but occasionally introduce variance (10% chance)
        if random.random() < 0.1:
            base_row['What is your course?'] = random.choice(courses)
        if random.random() < 0.1:
            base_row['Your current year of Study'] = random.choice(years)
        if random.random() < 0.1:
            base_row['What is your CGPA?'] = random.choice(cgpas)
            
        # Strongly maintain symptom correlation
        combo = random.choice(symptom_combos)
        for idx, col in enumerate(symptom_cols):
            base_row[col] = combo[idx]
        
        # Timestamp variation
        base_row['Timestamp'] = f"{random.randint(1,28)}/07/2020 {random.randint(0,23)}:{random.randint(10,59)}"
        
        augmented.append(base_row)
        
    return pd.DataFrame(augmented)

print("Starting expansion to ~1,000 records...")
new_females = generate_data(females, 500)
new_males = generate_data(males, 500)

expanded_df = pd.concat([new_females, new_males], ignore_index=True)

# Final NaN check for raw data
expanded_df['Age'] = expanded_df['Age'].fillna(expanded_df['Age'].median())
expanded_df = expanded_df.dropna()

# Shuffle
expanded_df = expanded_df.sample(frac=1).reset_index(drop=True)

expanded_df.to_csv(file_path, index=False)
print(f"Dataset expanded to {len(expanded_df)} records.")
