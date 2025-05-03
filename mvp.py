import streamlit as st
from genai import draft_future_complaint
import os


if "init" not in st.session_state:
    st.session_state.init = True


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
        st.write("Churn prediction:")
        st.write("TODO")
        

st.markdown(
    """
    ---

    ### ML-based churn prediction

    Load the csv file of a new customer to get a churn prediction based on a classification model  
    trained on other customers' data.  
    The model will also provide a lift analysis ...

    **Note:** The csv file should follow the format of this sample file:
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