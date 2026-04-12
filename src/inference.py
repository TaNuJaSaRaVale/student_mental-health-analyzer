import os
import pickle
import pandas as pd

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'mental_health_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

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
    model = load_model()
    
    # Needs to match the columns expected by the trained model (from X in data_processing.py)
    # The columns in the trained X were:
    # 'Choose your gender', 'Age', 'What is your CGPA?', 'Marital status', 
    # 'Do you have Anxiety?', 'Do you have Panic attack?', 
    # 'Did you seek any specialist for a treatment?'
    
    df_input = pd.DataFrame([{
        'Choose your gender': input_data['gender'],
        'Age': input_data['age'],
        'What is your CGPA?': input_data['cgpa'],
        'Marital status': input_data['marital_status'],
        'Do you have Anxiety?': input_data['anxiety'],
        'Do you have Panic attack?': input_data['panic_attack'],
        'Did you seek any specialist for a treatment?': input_data['treatment']
    }])
    
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1] # Probability of '1' (Depression)
    
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
