import os
import pickle
import pandas as pd

def load_model_and_scaler():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'mental_health_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict(input_data: dict) -> dict:
    """"
    Expects input_data dict with keys:
    - gender: int (0 for Male, 1 for Female)
    - age: float
    - cgpa: float
    - marital_status: int (0 or 1)
    - anxiety: int (0 or 1)
    - panic_attack: int (0 or 1)
    - treatment: int (0 or 1)
    """""
    model, scaler = load_model_and_scaler()
    
    # 1. Feature Engineering
    symptom_severity = input_data['anxiety'] + input_data['panic_attack']
    
    # Needs to match the columns expected by the trained model exactly
    df_input = pd.DataFrame([{
        'Choose your gender': input_data['gender'],
        'Age': input_data['age'],
        'What is your CGPA?': input_data['cgpa'],
        'Marital status': input_data['marital_status'],
        'Do you have Anxiety?': input_data['anxiety'],
        'Do you have Panic attack?': input_data['panic_attack'],
        'Did you seek any specialist for a treatment?': input_data['treatment'],
        'Symptom_Severity': symptom_severity
    }])
    
    # 2. Scale
    X_scaled = scaler.transform(df_input)
    
    # 3. Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1] # Probability of '1' (Depression)
    
    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "has_depression": bool(prediction == 1)
    }

if __name__ == "__main__":
    sample_input = {
        "gender": 1,
        "age": 20,
        "cgpa": 3.75,
        "marital_status": 0,
        "anxiety": 1,
        "panic_attack": 0,
        "treatment": 0
    }
    result = predict(sample_input)
    print("Sample Prediction:", result)
