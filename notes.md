# Task

## Requirements

1. ML: predictive model using structured data to identify customers at risk of churns and churn drivers
2. Gen-AI: extract insights to complemet the ML model (e.g. summaize common themes)

It is important to extract business insights from these.

## Deliverables

1. report: model implementations, pros and cons, scalability and costs
2. presentation: insights

---

# Plan

## Dataset headers:

- customerID
- gender
- SeniorCitizen: whether the customer is a senior (usually from 65 onwards) 
- Partner: whether the customer has a partner
- Dependents: whether the customer has people who depdend on him (children, relatives, etc)
- tenure: for how long
- PhoneService: whether the phone service is active
- MultipleLines: whether there are multiple lines for the phone service
- InternetService: type of internet service
- OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies: adds-on for the internet service
- Contract: type of contract
- PaperlessBilling: whether the customer receives paperless billing
- PaymentMethod: method of payment
- MonthlyCharges
- TotalCharges: blank values here with 0 tenure these are new customers
- Churn

## Data structured part (simple ML)

- Data processing: usual steps (missing values, encoding, scaling, etc)
- Feature engineering:
    - internet service security: feature indicating whether the customer has one of OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
    - streaming: feature indicating whether the customer has one of StreamingTV, StreamingMovies
    - contract stability: feature indicating whether the customer has a monthly or yearly (1 or 2) contract
    - financial strain: MonthlyCharges / tenure to capture cutomer's strain
    - discrepancy between expected and actual annual charges: TotalCharges - MonthlyCharges * tenure
    Potentially we could divide this over the tenure to capture the discrepancy monthly
    - complex household: feature indicating whether the customer has a partner or dependents
    - payment vulnerability: feature indicating whether the customer is safe
    (non-automatic payment methods vs automatic ones)
    - potentially combine the engineered features to create others
- Feature selection: use binary plots, correlation matrix, and feature importance with a simple model to filter features
- Potentially deal with class imbalance
- Model selection: usual suspects (logistic regression, random forest, gradient boosting, etc)
- Model evaluation: usual suspects (accuracy, precision, recall, f1, roc-auc, etc)

## Gen AI part

- cluster complaint themes
- perform sentiment analysis regarding the various themes and issues (adds-on, tech support, etc)
Then combine the results with the predictions from the ML model and the feature importance analys to identify the main churn drivers
(also consider somehow the number of complaints)
- Link the 2 tables to have more specific insight about churn drivers and then generalize to the whole dataset
- potentially use NLP to extract insights from the complaints

## Business insights

- Convert the findings into business insights
- Complaint resolution simulation: use gen AI to draft resolution scripts to complaints and recommend communication approaches
- Implement a feedback loop track effectiveness and refine recommendation strategies

## Timeline

1. Data processing and feature engineering (till April 6th)
- Data exploration
- Feature engineering and selection
- Address imabalanceness
- Start model selection and experimentation

2. Focused Development (till April 20th)
- ML Track: model development, evaluation, and refinement
- GenAI Track: complaint analysis, sentiment analysis, and understanding churn drivers 

3. Integration and Refinement (till April 27th)
- Correlate ML predictions and GenAI insights
- Attempt integration of insights with ML predictions when possible

4. Final Preparation (till May 6th)
- Convert findings into business insights
- Finalize presentation materials

---


# Development

## ML part

### Data processing and feature engineering
- The dataset has no missing values

## Gen AI
