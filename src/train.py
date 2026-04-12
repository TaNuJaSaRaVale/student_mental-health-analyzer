import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from data_processing import clean_data, get_features_and_target

def train_and_export_model():
    # Load Data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'Student Mental health.csv')
    df = pd.read_csv(data_path)
    
    # Process Data
    df_clean = clean_data(df)
    X, y = get_features_and_target(df_clean)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define Pipeline to FIX DATA LEAKAGE
    # StandardScaler now is part of the Pipeline, ensuring it ONLY fits on the training data!
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])
    
    # Hyperparameter Tuning using GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 10, None],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best Parameters Found:")
    print(grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test)
    print("\nTest Set Accuracy:", accuracy_score(y_test, y_pred))
    print("Test Set F1-Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Export Model
    model_export_path = os.path.join(os.path.dirname(__file__), '..', 'mental_health_model.pkl')
    with open(model_export_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    print(f"\nModel strictly exported to: {model_export_path}")

if __name__ == "__main__":
    train_and_export_model()
