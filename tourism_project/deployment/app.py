import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Monikasulur/tour", filename="tour_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
# User input
st.set_page_config(page_title="Wellness Tourism Package Prediction", layout="centered")

st.title("🌿 Wellness Tourism Package Prediction")

st.markdown("""
This application predicts the likelihood of a customer purchasing the **Wellness Tourism Package**
based on their demographic details and interaction history.

Please enter the customer information below to get a prediction.
""")

# --------------------------------------------------
# Input Fields
# --------------------------------------------------

st.header("Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

citytier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Freelancer", "Small Business", "Large Business", "Retired"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

numberofpersonvisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1, max_value=10, value=2
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

numberoftrips = st.number_input(
    "Number of Trips per Year",
    min_value=0, max_value=50, value=2
)

passport = st.selectbox(
    "Passport Available",
    ["No", "Yes"]
)

owncar = st.selectbox(
    "Owns a Car",
    ["No", "Yes"]
)

numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0, max_value=5, value=0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=5000, max_value=500000, value=30000
)

st.header("Customer Interaction Details")

pitchsatisfactionscore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1, max_value=5, value=3
)

productpitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "Wellness"]
)

numberoffollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0, max_value=10, value=2
)

durationofpitch = st.number_input(
    "Duration of Pitch (minutes)",
    min_value=1, max_value=60, value=15
)

# --------------------------------------------------
# Encode Inputs (Must match training encoding)
# --------------------------------------------------

input_dict = {
    "Age": age,
    "TypeofContact": 1 if typeofcontact == "Self Inquiry" else 0,
    "CityTier": citytier,
    "Occupation": ["Salaried", "Freelancer", "Small Business", "Large Business", "Retired"].index(occupation),
    "Gender": 1 if gender == "Male" else 0,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": ["Single", "Married", "Divorced"].index(maritalstatus),
    "NumberOfTrips": numberoftrips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if owncar == "Yes" else 0,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": ["Executive", "Manager", "Senior Manager", "AVP", "VP"].index(designation),
    "MonthlyIncome": monthlyincome,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "ProductPitched": ["Basic", "Standard", "Deluxe", "Super Deluxe", "Wellness"].index(productpitched),
    "NumberOfFollowups": numberoffollowups,
    "DurationOfPitch": durationofpitch
}

input_df = pd.DataFrame([input_dict])

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ The customer is **LIKELY** to purchase the Wellness Tourism Package.")
    else:
        st.warning("❌ The customer is **NOT LIKELY** to purchase the Wellness Tourism Package.")
