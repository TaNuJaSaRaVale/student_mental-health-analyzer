import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("mental_health_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="Mental Health Analyzer", page_icon="🧠")

st.title("🎓 Student Mental Health Analyzer")
st.markdown("### Enter student details to assess mental health risk")

st.markdown("---")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 15, 30, 20)
cgpa = st.slider("CGPA", 0.0, 4.0, 3.0)

marital = st.selectbox("Marital Status", ["No", "Yes"])
anxiety = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
panic = st.selectbox("Do you have Panic Attack?", ["No", "Yes"])
treatment = st.selectbox("Treatment Taken?", ["No", "Yes"])

st.markdown("---")

# Convert to numeric
gender_val = 1 if gender == "Female" else 0
marital_val = 1 if marital == "Yes" else 0
anxiety_val = 1 if anxiety == "Yes" else 0
panic_val = 1 if panic == "Yes" else 0
treatment_val = 1 if treatment == "Yes" else 0

# Predict
if st.button("Predict"):

    input_data = np.array([[gender_val, age, cgpa, marital_val, anxiety_val, panic_val, treatment_val]])

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)

    depression_prob = prob[0][1] * 100

    st.subheader("📊 Prediction Result")

    # Risk levels
    if depression_prob < 40:
        risk = "Low"
        st.success(f"✅ Low Risk ({depression_prob:.2f}%)")
    elif depression_prob < 70:
        risk = "Medium"
        st.warning(f"⚠️ Medium Risk ({depression_prob:.2f}%)")
    else:
        risk = "High"
        st.error(f"🚨 High Risk ({depression_prob:.2f}%)")

    # Progress bar
    st.progress(int(depression_prob))

    st.markdown("---")

    # Analysis
    st.subheader("🧠 Analysis")

    reasons = []

    if anxiety_val == 1:
        reasons.append("Presence of anxiety increases mental health risk")

    if panic_val == 1:
        reasons.append("Panic attacks are strongly linked with stress")

    if cgpa < 2.5:
        reasons.append("Low CGPA may indicate academic pressure")

    if marital_val == 1:
        reasons.append("Marital responsibilities may add stress factors")

    if age > 25:
        reasons.append("Higher age group shows slightly increased risk")

    if len(reasons) == 0:
        reasons.append("No major risk factors detected")

    for r in reasons:
        st.write("•", r)

    st.markdown("---")

    # Suggestions
    st.subheader("💡 Suggestions")

    if risk == "High":
        st.write("• Consider seeking professional help")
        st.write("• Talk to a trusted person")
        st.write("• Reduce workload and stress triggers")

    elif risk == "Medium":
        st.write("• Practice stress management techniques")
        st.write("• Maintain proper sleep schedule")
        st.write("• Take regular breaks")

    else:
        st.write("• Maintain your current healthy lifestyle")
        st.write("• Stay physically active")

    st.markdown("---")

    st.info("⚠️ This is a prediction system, not a medical diagnosis.")