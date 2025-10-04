
# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Load trained pipeline
# -------------------------------
@st.cache_resource
def load_model():
    # Load the model from the correct path
    return joblib.load("regimen_effectiveness_pipeline.pkl")

model = load_model()

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Regimen Effectiveness Predictor",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar decoration
# -------------------------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2966/2966485.png",
        width=120
    )
    st.markdown(
        """
        ## 💡 About
        This app predicts whether a patient
        on a given ART **regimen** is likely
        to achieve **viral suppression**.

        Upload patient data or use the form
        below to test the model.
        """
    )
    st.markdown("---")
    st.markdown("👨‍⚕️ **Built with ❤️ for HIV care programs**")

# -------------------------------
# Main Title
# -------------------------------
st.title("💊 ART Regimen Effectiveness Predictor")
st.markdown(
    "Enter patient details below and click **Predict** "
    "to see the likelihood of viral suppression."
)

# -------------------------------
# Input form
# -------------------------------
with st.form("prediction_form"):
    # Define all features used by the model - dynamically generated from the notebook's feature_names
    # Note: This list should match the 'feature_names' variable generated in the notebook
    # and will be used to create the input DataFrame with the correct columns.
    # Updated based on the output of cell 181ad0c1
    all_features = ['Age at reporting', 'Weight', 'Height', 'Last WHO Stage',
                    'Months of Prescription', 'time_on_art_days', 'Current Regimen',
                    'Sex', 'DOB', 'Last WHO Stage Date', 'Last VL Date',
                    'Last Visit Date', 'Active in TB', 'AHD Client']

    input_data = {}
    col1, col2 = st.columns(2)

    # Categorical features with limited options
    categorical_limited = ['Current Regimen', 'Sex', 'Active in TB', 'AHD Client']
    # Numeric features
    numeric_features = ['Age at reporting', 'Weight', 'Height', 'Last WHO Stage', 'Months of Prescription', 'time_on_art_days']
    # Date features (handle as strings for now, or add date picker logic)
    date_features = ['DOB', 'Last WHO Stage Date', 'Last VL Date', 'Last Visit Date']


    with col1:
        # Input fields for a subset of features
        input_data['Current Regimen'] = st.selectbox("Current Regimen", ["TDF/3TC/DTG", "ABC+3TC+DTG", "AZT+3TC/EFV", "Other", "Missing"]) # Added "Missing" to handle potential data issues
        input_data['Sex'] = st.radio("Sex", ["M", "F", "Missing"]) # Added "Missing"
        input_data['Age at reporting'] = st.number_input("Age (years)", min_value=1, max_value=100, value=30)
        input_data['Weight'] = st.number_input("Weight (kg)", min_value=10, max_value=200, value=65)
        input_data['Height'] = st.number_input("Height (cm)", min_value=50, max_value=250, value=165)
        input_data['Last WHO Stage'] = st.number_input("Last WHO Stage", min_value=1, max_value=4, value=1)


    with col2:
        input_data['Months of Prescription'] = st.number_input("Months of Prescription", min_value=0, max_value=12, value=3)
        input_data['time_on_art_days'] = st.number_input("Time on ART (days)", min_value=0, max_value=10000, value=365)
        input_data['Active in TB'] = st.radio("Active in TB", ["Yes", "No", "Missing"]) # Added "Missing"
        input_data['AHD Client'] = st.radio("AHD Client", ["Yes", "No", "Missing"]) # Added "Missing"

        # Date inputs (using text input for simplicity, could use st.date_input)
        input_data['DOB'] = st.text_input("Date of Birth (e.g., 01/01/1990)", "01/01/1990")
        input_data['Last WHO Stage Date'] = st.text_input("Last WHO Stage Date (e.g., 01/01/2024)", "01/01/2024")
        input_data['Last VL Date'] = st.text_input("Last VL Date (e.g., 01/01/2024)", "01/01/2024")
        input_data['Last Visit Date'] = st.text_input("Last Visit Date (e.g., 01/01/2024)", "01/01/2024")


    submitted = st.form_submit_button("🔮 Predict")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    # Prepare input DataFrame with ALL required columns, matching the order of training data
    # Create a dictionary with all features, using input_data and adding default/missing for others if needed
    # Ensure all_features list is exhaustive and matches the model's expectation
    full_input_data = {}
    # Populate with values from the form
    for feature in all_features:
        if feature in input_data:
             full_input_data[feature] = input_data[feature]
        else:
            # Handle features not in the form if any (shouldn't happen with the current all_features list)
            full_input_data[feature] = np.nan # Or a sensible default

    input_df = pd.DataFrame([full_input_data])

    # run prediction
    try:
        prob = model.predict_proba(input_df)[0, 1]
        label = model.predict(input_df)[0]

        # show results nicely
        st.success(f"✅ Prediction complete!")

        if label == 1:
            st.markdown(
                f"### 🎉 Likely Suppressed\n"
                f"The model predicts **viral suppression** with **{prob*100:.1f}% confidence**."
            )
        else:
            st.markdown(
                f"### ⚠️ Not Suppressed\n"
                f"The model predicts **no suppression** with only **{prob*100:.1f}% confidence**."
            )

        st.progress(float(prob))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("⚡ Powered by Streamlit + Machine Learning | Designed for HIV Program Analytics")
