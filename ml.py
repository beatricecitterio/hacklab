import pandas as pd
import numpy as np
import re
import warnings
import pickle
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE

# Default retention cost parameters (will be overridden by values from stl.py)
retention_factor = 0.05
retention_period = 6

# Functions for data processing
def preprocess_addson(df):
    """
    Set to 0 all the adds-on services for clients that have not bought the main service 
    (PhoneService and InternetService)
    """
    new_df = df.copy()
    cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    service_pattern = re.compile(r".*service.*", re.IGNORECASE)
    for col in cols:
        if col in new_df.columns:  # Check if column exists
            new_df[col] = new_df[col].apply(lambda x: "No" if isinstance(x, str) and service_pattern.match(x) else x)

    return new_df

def convert_boolean_variables(df):
    """Convert all categorical variables with two possible values into booleans."""
    new_df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() == 2:
            # Map the two unique values to 1 and 0
            unique_values = df[col].unique()
            new_df[col] = new_df[col].map({unique_values[0]: 0, unique_values[1]: 1})

    return new_df

def simplify_services(df):
    """Create engineered features from existing columns."""
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Check required columns exist before operations
    required_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
                    "StreamingTV", "StreamingMovies", "Partner", "Dependents", 
                    "Contract", "MonthlyCharges", "tenure", "TotalCharges", "PaymentMethod"]
    
    for col in required_cols:
        if col not in df.columns:
            if col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]:
                df[col] = "No"
            elif col in ["StreamingTV", "StreamingMovies"]:
                df[col] = "No"
            elif col in ["Partner", "Dependents"]:
                df[col] = "No"
            elif col == "Contract":
                df[col] = "Month-to-month"
            elif col == "PaymentMethod":
                df[col] = "Credit card (automatic)"
    
    # Now proceed with feature engineering
    df["InternetSecurity"] = df[["OnlineSecurity", "OnlineBackup", "DeviceProtection"]].apply(
        lambda row: 'Yes' if 'Yes' in row.values else 'No', axis=1)
    df["Streaming"] = df[["StreamingTV", "StreamingMovies"]].apply(
        lambda row: 'Yes' if 'Yes' in row.values else 'No', axis=1)
    df["HouseholdComplexity"] = df[["Partner", "Dependents"]].apply(
        lambda x: sum(val == 'Yes' for val in x), axis=1)
    
    # Handle TotalCharges more carefully
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns and "tenure" in df.columns:
        df["ChargesDiscrepancy"] = df["MonthlyCharges"] * df["tenure"] - df["TotalCharges"]
    else:
        df["ChargesDiscrepancy"] = 0
    
    # Safe mapping for Contract
    if "Contract" in df.columns:
        contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        df["ContractStability"] = df["Contract"].map(lambda x: contract_map.get(x, 0))
    else:
        df["ContractStability"] = 0
    
    df["TechSupport_InternetSecurity"] = df[["TechSupport", "InternetSecurity"]].apply(
        lambda x: 'Yes' if 'Yes' in x.values else 'No', axis=1)
    
    # Handle SeniorCitizen mapping
    if "SeniorCitizen" in df.columns:
        df["Senior"] = df["SeniorCitizen"].map({1: "Yes", 0: "No"})
    else:
        df["Senior"] = "No"
    
    # Financial strain calculation
    if "MonthlyCharges" in df.columns and "tenure" in df.columns:
        # Avoid division by zero
        df["FinancialStrain"] = df.apply(
            lambda row: row["MonthlyCharges"] / row["tenure"] if row["tenure"] > 0 else row["MonthlyCharges"], 
            axis=1
        )
    else:
        df["FinancialStrain"] = 0
    
    # Payment safety feature
    if "PaymentMethod" in df.columns:
        automatic_pattern = re.compile(r"\bautomatic\b", re.IGNORECASE)
        df["PaymentSafety"] = df["PaymentMethod"].apply(
            lambda x: 1 if isinstance(x, str) and automatic_pattern.search(x) else 0
        )
    else:
        df["PaymentSafety"] = 0
    
    df["FullInternetSecurity"] = df[["InternetSecurity", "TechSupport_InternetSecurity"]].apply(
        lambda x: 'Yes' if 'Yes' in x.values else 'No', axis=1)
    
    return df

def process_data(df):
    """Process the dataset for modeling."""
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Check required columns exist
    required_cols = ["tenure", "TotalCharges", "MonthlyCharges"]
    for col in required_cols:
        if col not in df.columns:
            if col == "tenure":
                df[col] = 1  # Default value
            elif col == "TotalCharges":
                df[col] = df["MonthlyCharges"] if "MonthlyCharges" in df.columns else 0
            elif col == "MonthlyCharges":
                df[col] = 0  # Default value
    
    # Remove new customers and handle missing values
    df = df[df['tenure'] != 0]  # remove new customers
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    
    # Feature Engineering
    df = simplify_services(df)
    
    # Bin continuous features
    if "MonthlyCharges" in df.columns:
        df["MonthlyChargesBin"] = pd.cut(df["MonthlyCharges"], bins=[0, 35, 70, 105, 140], labels=False)
    else:
        df["MonthlyChargesBin"] = 0
        
    if "tenure" in df.columns:
        df["TenureBin"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=False)
    else:
        df["TenureBin"] = 0
    
    # Preprocess add-on services
    df = preprocess_addson(df)
    
    # Convert boolean variables
    df = convert_boolean_variables(df)
    
    return df

def encode_features(df):
    """Encode categorical features for model training."""
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Encode remaining categorical variables
    for col in df.columns:
        if df[col].dtype == "object" and df[col].nunique() > 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    # One-hot encoding
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def predict_churn(upload_file, st=None):
    """
    Main function to process data and make predictions.
    
    Parameters:
        upload_file: The uploaded Excel file containing customer data
        st: The Streamlit module, if None, no Streamlit-specific operations will be performed
    
    Returns:
        DataFrame containing predictions and analysis
    """
    try:
        # Load the uploaded data
        try:
            clients = pd.read_excel(upload_file, index_col=0)

        except Exception as e:
            if st:
                st.error(f"Error loading file: {e}")
            print(f"Error loading file: {e}")
            return None
        
        # # Keep a copy of the original data
        # original_clients = clients.copy()
        
        # Data processing
        clients = process_data(clients)
        client_ids = clients.index.tolist()
        
        # Calculate average customer lifetime value for later use
        average_monthly_revenue = clients["MonthlyCharges"].mean()
        average_tenure_months = clients["tenure"].mean()
        customer_lifetime_value = average_monthly_revenue * average_tenure_months
        
        # Initialize variables for use in both branches
        threshold = 0.5  # Default threshold
        X = None
        
        # Split target and features
        if 'Churn' in clients.columns:
            # For training data (with known churn status)
            y = clients['Churn']
            X = clients.drop(columns=['Churn'])
            
            # Encode features
            X = encode_features(X)
            
            # Load pre-trained model parameters from file logistic_model.pkl
            logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
            
            # Make predictions
            churn_probabilities = logistic_model.predict_proba(X)[:, 1]
            
            # Calculate optimal threshold using ROC curve
            fpr, tpr, thresholds = roc_curve(y, churn_probabilities)
            gmeans = np.sqrt(tpr * (1 - fpr))
            threshold_ix = np.argmax(gmeans)
            threshold = thresholds[threshold_ix]
        else:
            # For prediction data (without known churn status)
            if st:
                st.warning("No 'Churn' column found. Using pre-trained model parameters.")
            
            # Encode features
            X = encode_features(clients)
            
            # Load pre-trained model parameters from file logistic_model.pkl
            logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
            
            # Make predictions
            churn_probabilities = logistic_model.predict_proba(X)[:, 1]
        
        # Train retention cost model using linear regression with the user-specified parameters
        cost_features = ["MonthlyCharges", "tenure", "ContractStability"]
        
        # Use the global parameters (which can be overridden by stl.py)
        global retention_factor, retention_period
        clients["retention_cost_est"] = retention_factor * clients["MonthlyCharges"] * retention_period
        
        
        cost_reg = LinearRegression()
        cost_reg.fit(clients[cost_features], clients["retention_cost_est"])
        
        # Calculate retention costs
        retention_costs = cost_reg.predict(clients[cost_features])
        
        # Calculate predicted churn based on threshold
        predicted_churn = (churn_probabilities > threshold).astype(int)
        
        # Segment customers by risk level
        risk_levels = pd.cut(
            churn_probabilities, 
            bins=[0, 0.25, 0.5, 0.75, 1.0], 
            labels=["Low", "Medium", "High", "Critical"]
        )
        
        # Initialize profits
        profits = pd.Series(0.0, index=clients.index)
        
        # Calculate profit from using the model
        if 'Churn' in clients.columns:
            # For training data with known churn status
            actual_churn = y.values
            
            # False positives: predicted to churn but didn't
            FP = ((predicted_churn == 1) & (actual_churn == 0)).sum()
            # False negatives: predicted not to churn but did
            FN = ((predicted_churn == 0) & (actual_churn == 1)).sum()
            # True positives: predicted to churn and did
            TP = ((predicted_churn == 1) & (actual_churn == 1)).sum()
            
            # Calculate costs and benefits
            false_positive_cost = sum(retention_costs[(predicted_churn == 1) & (actual_churn == 0)])
            false_negative_cost = FN * customer_lifetime_value
            true_positive_value = sum(customer_lifetime_value - retention_costs[(predicted_churn == 1) & (actual_churn == 1)])
            
            # Net benefit
            net_benefit = true_positive_value - false_positive_cost - false_negative_cost
            
            # True positives get profit = customer_lifetime_value - retention_cost
            mask_tp = (predicted_churn == 1) & (actual_churn == 1)
            if mask_tp.any():
                profits[mask_tp] = customer_lifetime_value - retention_costs[mask_tp]
            
            # False positives lose retention_cost
            mask_fp = (predicted_churn == 1) & (actual_churn == 0)
            if mask_fp.any():
                profits[mask_fp] = -retention_costs[mask_fp]
            
            # False negatives lose customer_lifetime_value
            mask_fn = (predicted_churn == 0) & (actual_churn == 1)
            if mask_fn.any():
                profits[mask_fn] = -customer_lifetime_value
            
            # True negatives have no financial impact in this simplified model
            
            if st:
                st.success(f"Net benefit of model: ${net_benefit:.2f}")
                st.write(f"Customer lifetime value: ${customer_lifetime_value:.2f}")
                st.write(f"Average retention cost: ${retention_costs.mean():.2f}")
            
            # Create results dataframe
            results = pd.DataFrame({
                'Customer_ID': client_ids,
                'Churn_Probability': churn_probabilities,
                'Risk_Segment': risk_levels,
                'Retention_Cost': retention_costs,
                'Profit': profits
            })
        else:
            # For prediction data without known churn status
            # In this case, we can't calculate actual profit since we don't know the true churn value
            if st:
                st.info("Cannot calculate actual profit without known churn values.")
                st.write(f"Average retention cost: ${retention_costs.mean():.2f}")

            # Create results dataframe
            results = pd.DataFrame({
                'Customer_ID': client_ids,
                'Churn_Probability': churn_probabilities,
                'Risk_Segment': risk_levels,
                'Retention_Cost': retention_costs
            })
        # Reset index to ensure no extra initial column of indices
        results.reset_index(drop=True, inplace=True)
        
        return results
        
    except Exception as e:
        if st:
            st.error(f"An error occurred: {e}")
        print(f"Error in predict_churn: {e}")
        import traceback
        traceback.print_exc()
        return None

# This allows the script to be run as a standalone Streamlit app
# but doesn't execute when imported as a module
if __name__ == "__main__":
    import streamlit as st
    
    # Page title and description
    st.title("Customer Churn Prediction")
    st.markdown("""
    This application predicts the likelihood of customers churning based on their data.
    Upload your customer data file (in Excel format) to get predictions.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Customer Data (Excel format)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        st.info("Processing file... This may take a moment.")
        
        # Process the file and get results
        results = predict_churn(uploaded_file, st)
        
        if results is not None:
            # Display results
            st.subheader("Churn Prediction Results")
            st.dataframe(results)
            
            # Download button for results
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )