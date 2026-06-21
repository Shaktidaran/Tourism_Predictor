
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# --- Configuration Constants ---
HF_REPO_ID = "Shaktidaran/TourismPredictor"
HF_MODEL_FILENAME = "best_tourism_predictor_model_v1.joblib"

# --- Helper function to load model ---
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Streamlit UI ---
st.set_page_config(page_title="Tourism Package Prediction", layout="centered")
st.title("🌴 Wellness Tourism Package Prediction ✈️")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package 
based on their demographic and interaction data. 
Please enter the customer details below to get a prediction.
""")

if model is None:
    st.stop()

# --- User Input Fields ---
with st.sidebar:
    st.header("Customer Details")

    age = st.slider("Age", 18, 80, 37)
    typeof_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited', 'Unknown'])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 150, 15)
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer', 'Other'])
    gender = st.selectbox("Gender", ['Female', 'Male', 'Prefer Not to Say'])
    number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 5, 2)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unknown'])
    number_of_trips = st.slider("Number of Trips Annually", 1, 30, 3)
    passport = st.checkbox("Has Passport")
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.checkbox("Owns Car")
    number_of_children_visiting = st.slider("Number of Children Visiting (<5 years)", 0, 3, 1)
    monthly_income = st.slider("Monthly Income", 1000, 100000, 23000)
    product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Other'])
    number_of_followups = st.slider("Number of Follow-ups", 1, 6, 3)
    designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'President', 'Other'])


# --- Prediction Button and Logic ---
if st.button("Predict Purchase"):
    # Assemble input into DataFrame
    input_data = pd.DataFrame([{
        'Age': age,
        'TypeofContact': typeof_contact,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': number_of_person_visiting,
        'PreferredPropertyStar': preferred_property_star,
        'MaritalStatus': marital_status,
        'NumberOfTrips': number_of_trips,
        'Passport': 1 if passport else 0, # Convert boolean to int
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'OwnCar': 1 if own_car else 0, # Convert boolean to int
        'NumberOfChildrenVisiting': number_of_children_visiting,
        'MonthlyIncome': monthly_income,
        'ProductPitched': product_pitched,
        'NumberOfFollowups': number_of_followups,
        'Designation': designation
    }])

    try:
        prediction_proba = model.predict_proba(input_data)[:, 1]
        prediction = (prediction_proba >= 0.50).astype(int) # Using the threshold from train.py

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"The model predicts: **YES**, the customer is likely to purchase the package! (Probability: {prediction_proba[0]:.2f})")
        else:
            st.info(f"The model predicts: **NO**, the customer is unlikely to purchase the package. (Probability: {prediction_proba[0]:.2f})")
        
        st.write("""
        *Note: This prediction is based on the provided inputs and the trained model. 
        Actual purchase behavior may vary.*
        """)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

