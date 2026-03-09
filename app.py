import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack

# Load saved model and vectorizer
svm_model = joblib.load("models/svm_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

st.title("Fake Job Posting Detector")

st.write("Enter job details to check if a posting might be fraudulent.")

# User input
description = st.text_area("Job Description")

company_profile_present = st.selectbox(
    "Is company profile provided?", ["Yes", "No"]
)

has_company_logo = st.selectbox(
    "Does the job posting have a company logo?", ["Yes", "No"]
)

# Prediction button
if st.button("Predict"):

    company_profile_missing = 0 if company_profile_present == "Yes" else 1
    has_company_logo = 1 if has_company_logo == "Yes" else 0

    text_features = tfidf.transform([description])

    structured_features = np.zeros((1, 16))

    structured_features[0,0] = company_profile_missing
    structured_features[0,1] = has_company_logo

    final_features = hstack([text_features, structured_features])

    prediction = svm_model.predict(final_features)[0]

    if prediction == 1:
        st.error("⚠️ This job posting may be FRAUDULENT")
    else:
        st.success("✅ This job posting appears LEGITIMATE")