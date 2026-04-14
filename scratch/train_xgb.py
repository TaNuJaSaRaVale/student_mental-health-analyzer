import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processing import clean_data

# 1. Load Data
df = pd.read_csv('Student Mental health.csv')

# Clean existing NaN age (from the raw data) before processing
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)

# Apply standard cleaning mapping
df = clean_data(df)

# 2. Feature Engineering
# Academic Stress Index: Lower CGPA + Higher Year = More Stress (Simplified representation)
# Course year is usually 1, 2, 3, 4. Since clean_data drops 'Your current year of Study', we need to retain it or compute before dropping.
# Wait, clean_data DROPS 'Your current year of Study'. Let's do a basic symptom severity instead.

df['Symptom_Severity'] = df['Do you have Depression?'] + df['Do you have Anxiety?'] + df['Do you have Panic attack?']

# 3. Prepare Data
X = df.drop('Do you have Depression?', axis=1) # Target is depression
y = df['Do you have Depression?']

# Note: We included Depression in Symptom_Severity, which is DATA LEAKAGE!
# We must ONLY engineer on other symptoms!
X = df.drop('Do you have Depression?', axis=1)
X['Symptom_Severity'] = X['Do you have Anxiety?'] + X['Do you have Panic attack?']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. XGBoost & Hyperparameter Tuning
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 5. Evaluate
y_pred = best_model.predict(X_test)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save Model and Scaler
pickle.dump(best_model, open("mental_health_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
print("Model and scaler saved successfully.")
