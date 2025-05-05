import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.genai import draft_future_complaint
import streamlit as st
import src.ml as ml
import pickle
import joblib

if "init" not in st.session_state:
    st.session_state.init = True
    st.session_state.results = None
    st.session_state.retention_factor = 0.05
    st.session_state.retention_period = 6


st.markdown(
    """
    ### Complaint analysis
    The complaint analysis currently supports two main features:

    ##### 1. Drafting Future Complaints:
    Input a complaint history to get a draft of the potential following complaint based on:
    - All complaints of the customer
    - Similar complaints from other customers

    ##### 2. Churn prediction:
    Input a complaint to get a prediction of the likelyhood of the customer churning  
    based on on a text-classification model trained on other complaints.
    """
)

do_draft = st.checkbox("Draft a future complaint", value=False)
do_churn = st.checkbox("Predict churn", value=False)

complaint = st.chat_input(placeholder="Insert the complaint here")
if complaint:
    if do_draft:
        next_complaint = draft_future_complaint(complaint)
        st.write("Drafted complaint:")
        st.write(next_complaint)
    if do_churn:
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        model = joblib.load("models/complaint_classifier.pkl")

        complaint_vector = vectorizer.transform([complaint])
        churn_prob = model.predict_proba(complaint_vector)[0][1]

        st.write(f"Predicted churn probability: **{churn_prob:.2%}**")


# ML-based churn prediction section
st.markdown(
    """
    ---

    ### ML-based churn prediction

    Load a file of customers (CSV or Excel) to get churn predictions based on a classification model  
    trained on customers' data.  
    The model will also provide risk segmentation and profit analysis.

    **Note:** The file should contain customer data with columns like those in the original dataset. Here you can find a sample file:
    """
)

with open(os.path.join(os.getcwd(), "sample.csv"), "r") as file:
    csv_content = file.read()
st.download_button(
    label="Sample file",
    data=csv_content,
    file_name="sample.csv",
    mime="text/csv"
)

# Retention cost parameters
st.subheader("Retention Cost Parameters")
col1, col2 = st.columns(2)
with col1:
    retention_factor = st.number_input(
        "Retention Cost Factor (percentage of monthly charges)",
        min_value=0.01,
        max_value=1.0,
        value=st.session_state.retention_factor,
        step=0.01,
        help="This factor multiplies the monthly charges to estimate retention cost (e.g., 0.05 = 5% of monthly charges)"
    )
with col2:
    retention_period = st.number_input(
        "Retention Period (months)",
        min_value=1,
        max_value=24,
        value=st.session_state.retention_period,
        step=1,
        help="Number of months over which retention costs are calculated"
    )

# Save parameter values to session state
st.session_state.retention_factor = retention_factor
st.session_state.retention_period = retention_period

# File uploader for ML-based prediction
uploaded_file = st.file_uploader("Upload Customer Data (Excel format)", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    
    # Override the retention cost calculation in ml.py by modifying the module
    ml.retention_factor = retention_factor
    ml.retention_period = retention_period
    
    # Process the file and get results
    results = ml.predict_churn(uploaded_file, st)
    st.session_state.results = results

    if results is not None:
        # Search functionality
        st.subheader("Search by Customer ID")
        customer_search = st.text_input("Enter Customer ID to search", "")
        
        if customer_search:
            # Filter results by customer ID (case-insensitive partial match)
            filtered_results = results[results["Customer_ID"].astype(str).str.contains(customer_search, case=False)]
            
            if not filtered_results.empty:
                st.write(f"Found {len(filtered_results)} matching customers:")
                st.dataframe(filtered_results, hide_index=True)  # Remove index when displaying
            else:
                st.warning(f"No customers found matching '{customer_search}'")
        
        # Display all results
        st.subheader("All Churn Prediction Results")
        st.dataframe(results, hide_index=True)
        
        # Download button for results
        csv = results.to_csv(index=False)  # Remove index when exporting
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )
