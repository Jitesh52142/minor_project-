# Import required libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Decision Tree model from a pickle file
# This model has been previously saved and can now be used for prediction in the app.
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions using the trained model
# This function takes in a list of features, which are pre-processed inputs.
# It feeds the features to the model and returns the prediction.
def make_prediction(features):
    # Since the model expects the input in the form of an array, we pass it as a 2D array
    # to match the expected input shape.
    prediction = model.predict([features])
    return prediction

# Streamlit UI configuration
# Set page title and icon for the web app
st.set_page_config(page_title="Women's Safety Risk Prediction", page_icon="🚨", layout="wide")

# Title of the app displayed at the top
st.title("🚨 Women's Safety Risk Prediction")

# Add a short description of the app functionality
st.write("""
    This app helps predict the safety risk in a location based on various factors such as:
    - Signal strength
    - Gender ratio (Male and Female count)
    - Crowd density
    - Time and date factors
    The prediction helps identify areas with potential safety concerns for women.
""")

# Display instructions for the users to follow
st.markdown("""
    **How to use the app:**
    1. Enter the required parameters such as tower ID, latitude, longitude, etc., in the sidebar.
    2. Click on the **Make Prediction** button to calculate the safety risk based on the inputs.
    3. View the prediction results and insights about the location’s safety.
""")

# Sidebar for user input - each input is explained below:
st.sidebar.header("Input Parameters")
# Tower ID input field for users to enter the unique identifier for the tower
# We use number_input to ensure that only integers are accepted for tower_id
tower_id = st.sidebar.number_input("Tower ID", min_value=1, step=1, format="%d")  # Tower ID as integer

# Latitude input field - allows users to enter the geographical latitude of the location
# The input value is constrained to valid latitude values: -90.0 to 90.0 degrees.
latitude = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.0001, format="%.4f")

# Longitude input field - allows users to enter the geographical longitude of the location
# The input value is constrained to valid longitude values: -180.0 to 180.0 degrees.
longitude = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.0001, format="%.4f")

# Signal Strength input field - users can enter signal strength in dBm (negative values)
# Typically, signal strength ranges from -120 dBm to 0 dBm, with 0 being the best signal.
signal_strength = st.sidebar.number_input("Signal Strength (dBm)", min_value=-120, max_value=0, step=1)

# Male Count input field - user inputs the number of males in the location
male_count = st.sidebar.number_input("Male Count", min_value=0, step=1)

# Female Count input field - user inputs the number of females in the location
female_count = st.sidebar.number_input("Female Count", min_value=0, step=1)

# Crowd Density input field - defines the crowd density as people per square meter
# Higher values could indicate a crowded area, potentially more risky.
crowd_density = st.sidebar.number_input("Crowd Density (people/m²)", min_value=0, step=1)

# Hour input field - users can select the hour of the day (0-23) to simulate time-based factors
hour = st.sidebar.slider("Hour", 0, 23)

# Day of Week input field - users can select the day of the week (0=Sunday, 6=Saturday)
day_of_week = st.sidebar.slider("Day of Week", 0, 6, help="0=Sunday, 6=Saturday")

# Month input field - users can select the month of the year (1-12)
month = st.sidebar.slider("Month", 1, 12)

# Weekend input radio button - allows the user to choose if it is a weekend (Yes/No)
is_weekend = st.sidebar.radio("Is Weekend?", ("Yes", "No"))

# Convert the weekend input to binary format, as the model expects a 0 or 1 for this feature
is_weekend = 1 if is_weekend == "Yes" else 0

# Organize the input data into a list (features array) to feed into the model
features = [tower_id, latitude, longitude, signal_strength, male_count, female_count, crowd_density, hour, day_of_week, month, is_weekend]

# A button that, when clicked, will make the prediction using the input features
if st.sidebar.button("Make Prediction"):
    # Display a spinner to show the app is working on the prediction
    with st.spinner('Making prediction...'):
        # Get the prediction from the model
        prediction = make_prediction(features)
        # Show the prediction result on the app
        st.success(f"Prediction: {prediction[0]}")

    # Display the input data that the user provided for reference
    st.write("### Input Data:")
    input_data = pd.DataFrame([features], columns=[
        "Tower ID", "Latitude", "Longitude", "Signal Strength (dBm)", 
        "Male Count", "Female Count", "Crowd Density (people/m²)", 
        "Hour", "Day of Week", "Month", "Is Weekend?"
    ])
    st.dataframe(input_data)  # Show the data as a table

    # Provide additional explanation of the prediction results
    st.write("""
    ### Explanation:
    - **Low risk**: A lower safety risk means the location is safer, based on the model’s prediction.
    - **High risk**: A higher risk means the location might be unsafe, requiring caution.
    """)

# Add a section to describe the model used
st.markdown("""
    ## Model Overview
    The model used in this application is a **Decision Tree Classifier**. It analyzes various input features such as:
    - Signal strength, crowd density, gender ratio, and time-based factors.
    The model classifies locations into high or low safety risk categories.
""")

# Display some helpful tips for users to interpret the data
st.markdown("""
    ## Helpful Tips:
    - **Signal Strength**: Low signal strength may indicate a more dangerous area due to weak connectivity.
    - **Crowd Density**: Crowded areas can be more risky, depending on the time of day.
    - **Gender Ratio**: Significant imbalances in male-to-female ratio may indicate a potential risk.
""")

# Footer with contact details and app information
st.markdown("""
    ## About This App
    This app was created to help assess the safety risks in various locations based on real-time data.
    It identifies areas where extra caution may be necessary.

    For inquiries or suggestions, you can reach us at:  
    [Contact Email](mailto:contact@womensafety.com)
""")
