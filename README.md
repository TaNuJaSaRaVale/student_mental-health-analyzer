# 🧠 Student Mental Health Analyzer (Machine Learning Project)

An end-to-end Machine Learning pipeline and full-stack web application designed to predict the likelihood of depression in university students based on academic, physiological, and lifestyle factors.

---

## 🎯 Project Objective & Vision

Student mental health is a critical, often under-reported issue in modern academia. The stress of coursework, combined with personal lifestyle factors, can silently lead to extreme psychological strain. 

**This project aims to detect potential mental health risks early.** By analyzing data points such as a student's age, CGPA, panic attack history, anxiety, and relationship status, the ML model provides a risk assessment to encourage students to proactively seek professional help.

---

## 🛠️ The Machine Learning Pipeline

To ensure the model is suitable for "real-world" inference, the project utilizes advanced Data Science methodologies rather than relying on basic algorithms. 

### 1. Robust Dataset Generation & Balancing
Starting from a foundational dataset of ~100 responses, an advanced data augmentation script was used to safely expand the dataset to **1,000 realistic records**. This process jittered continuous values (like Age) using statistical normal distributions and balanced the gender ratio (50% Male / 50% Female) to prevent the AI from developing gender biases.

### 2. Feature Engineering
Raw data is rarely enough. Our preprocessing pipeline introduces **`Symptom_Severity`**—an engineered feature that mathematically combines the presence of Anxiety and Panic Attacks. This grants the model deeper psychological context before it attempts to classify depression.

### 3. Advanced Modeling: XGBoost & GridSearchCV
Instead of basic algorithms, the engine relies on **XGBoost (Extreme Gradient Boosting)**, an industry-standard algorithm known for its high performance on tabular data.
To prevent overfitting:
- **GridSearchCV** was deployed to automatically test 54 combinations of hyperparameters (`max_depth`, `learning_rate`, `n_estimators`, `subsample`).
- **Stratified 5-Fold Cross-Validation** was used to ensure the model generalizes perfectly across unseen data splits, yielding an optimal and reliable real-world accuracy score (~75-80%).

### 4. End-to-End Application Infrastructure
The model (`model.pkl` and `scaler.pkl`) is served through robust logic that automatically calculates engineered features on the fly, making it scalable for frontend integration.

---

## 💻 Tech Stack
*   **AI/ML Core**: Python, XGBoost, Scikit-Learn, Pandas, NumPy
*   **API / Backend**: FastAPI, Uvicorn, Pydantic
*   **Data Visualization**: Matplotlib, Seaborn
*   **Web Interfaces**: 
    1. Streamlit Dashboard (Rapid prototyping UI)
    2. React/Vite Frontend (Professional user interface)

---

## 🚀 How to Run the Project Locally

Because this project features both a pure Python backend API and a modern JS frontend, you will need two terminal windows.

### Terminal 1: Start the Backend AI Engine
1. Open terminal in the root folder (`mental-health-analyzer`).
2. Install the necessary Python packages: `pip install -r requirements.txt`
3. Launch the API:
   ```bash
   uvicorn api.index:app --reload
   ```

### Terminal 2: Start the Web UI
1. Open a new terminal and navigate to the frontend folder:
   ```bash
   cd frontend
   ```
2. Start the React server:
   ```bash
   npm run dev
   ```
*(A local link like `http://localhost:5173` will be generated. Open this in your browser to interact with the model.)*

*(Optional: To run the secondary Python-only Streamlit interface, simply run `streamlit run app.py` in the root folder).*

---

## ⚠️ Important Analytical Disclaimer Regarding Data Logic

> **Note on Marital Status and Student Stress:**
> During the evaluation of the model's logic, statistical correlations may appear regarding marital status. **This model does *not* state that marriage generally causes high levels of depression in the real world.** Rather, it specifies that *students* who are married—while simultaneously attempting to manage university courses, potential financial strain, and examinations—exhibit statistically higher rates of psychological distress due to combining heavy academic burdens with significant personal responsibilities.
> 
> *Furthermore, this application is a predictive academic tool. It is not a medical diagnosis tool and should never be used as a substitute for certified psychiatric evaluation.*

---

