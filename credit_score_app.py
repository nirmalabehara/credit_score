import streamlit as st
import pandas as pd
import joblib  

# Load the trained model & encoders
model = joblib.load("credit_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Custom CSS for full-screen layout
st.markdown(
    """
    <style>
    html {
       background: linear-gradient(135deg, #007BFF, #00d4ff); 
    }
    unsafe_allow_html=True

    .main-container {
    background: none;
    box-shadow: none;
    padding: 10px;
    width: 100vw;
    height: 0vh;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #007BFF;
        color: white;
        font-size: 16px;
        padding: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .prediction-container {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    .good-credit {
        background-color: #d4edda;
        color: #155724;
    }
    .bad-credit {
        background-color: #f8d7da;
        color: #721c24;
    }
    hr {
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main container for full-screen layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Title & Subtitle
st.markdown("<h2 style='color:#007BFF; text-align:center;'>Credit Scoring Prediction</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Enter details below to assess creditworthiness</p>", unsafe_allow_html=True)

# Use two-column layout for input fields
col1, col2 = st.columns(2)

# Input fields in two columns for clean layout
with col1:
    Status = st.selectbox("Status", ["A11", "A12", "A13", "A14"])
    Duration = st.slider("Duration (months)", 1, 72, 12)
    CreditHistory = st.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"])
    Purpose = st.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A44"])
    CreditAmount = st.number_input("Credit Amount", min_value=100, max_value=20000, step=100)
    Savings = st.selectbox("Savings", ["A61", "A62", "A63", "A64", "A65"])
    Employment = st.selectbox("Employment", ["A71", "A72", "A73", "A74", "A75"])

with col2:
    InstallmentRate = st.slider("Installment Rate", 1, 4, 2)
    SexMarital = st.selectbox("Sex & Marital Status", ["A91", "A92", "A93", "A94"])
    Guarantors = st.selectbox("Guarantors", ["A101", "A102", "A103"])
    ResidenceDuration = st.slider("Residence Duration", 1, 4, 2)
    Property = st.selectbox("Property", ["A121", "A122", "A123", "A124"])
    Age = st.slider("Age", 18, 75, 30)
    Housing = st.selectbox("Housing", ["A151", "A152", "A153"])

# Second row for remaining fields
col3, col4 = st.columns(2)

with col3:
    ExistingCredits = st.slider("Existing Credits", 1, 4, 1)
    Job = st.selectbox("Job", ["A171", "A172", "A173", "A174"])
    NumPeopleLiable = st.slider("Number of People Liable", 1, 2, 1)

with col4:
    Telephone = st.selectbox("Telephone", ["A191", "A192"])
    ForeignWorker = st.selectbox("Foreign Worker", ["A201", "A202"])
    OtherInstallments = st.selectbox("Other Installments", ["A141", "A142", "A143"])

# Divider
st.markdown("<hr>", unsafe_allow_html=True)

# Prepare input data
data = {
    "Status": [Status], "Duration": [Duration], "CreditHistory": [CreditHistory],
    "Purpose": [Purpose], "CreditAmount": [CreditAmount], "Savings": [Savings],
    "Employment": [Employment], "InstallmentRate": [InstallmentRate], "SexMarital": [SexMarital],
    "Guarantors": [Guarantors], "ResidenceDuration": [ResidenceDuration], "Property": [Property],
    "Age": [Age], "OtherInstallments": [OtherInstallments], "Housing": [Housing],
    "ExistingCredits": [ExistingCredits], "Job": [Job], "NumPeopleLiable": [NumPeopleLiable],
    "Telephone": [Telephone], "ForeignWorker": [ForeignWorker]
}

user_input = pd.DataFrame(data)

# Encode categorical variables
for col in user_input.select_dtypes(include=["object"]).columns:
    user_input[col] = label_encoders[col].transform(user_input[col])

# Scale numerical values
user_input[["Duration", "CreditAmount", "Age"]] = scaler.transform(user_input[["Duration", "CreditAmount", "Age"]])

# Predict only when button is clicked
if st.button("Predict Credit Score"):
    prediction = model.predict(user_input)
    result = "Good Credit" if prediction[0] == 0 else "Bad Credit"
    
    # Apply styling based on prediction
    result_class = "good-credit" if prediction[0] == 0 else "bad-credit"
    
    # Display result with professional styling
    st.markdown(f"<div class='prediction-container {result_class}'>{result}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # Close the main container
