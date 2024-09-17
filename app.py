import streamlit as st
import pickle
import numpy as np

# Load the model and data
df = pickle.load(open('Crop-price.df.pkl', 'rb'))
pipe = pickle.load(open('pred_model2_this.pkl', 'rb'))

st.set_page_config(
    page_title="Crop Price Predictor",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Crop Price Predictor")

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    N_SOIL = st.number_input('N_Soil Value: ', min_value=10, step=1)
    P_SOIL = st.number_input('P_Soil Value: ', min_value=10, step=1)
    Temp = st.number_input('Temperature: ', min_value=-10, step=1)
    Humidity = st.number_input('Humidity: ', min_value=1, step=1)
    ph = st.number_input('PH value: ', min_value=1, step=1)
    Rainfall = st.number_input('RainFall: ', min_value=1, step=1)
    State = st.selectbox('State: ', list(df['STATE'].unique()))
    Crop = st.selectbox('Crop: ', list(df['CROP'].unique()))
import random

if st.button('Predict Price'):
    try:
        # Ensure all fields are selected
        if "None" in [N_SOIL, P_SOIL, Temp, Humidity, ph, Rainfall, State, Crop]:
            st.error("Please fill out all fields before predicting.")
        else:
            # Convert categorical variables (State, Crop) to one-hot encoded values
            state_onehot = np.zeros(len(df['STATE'].unique()))  # One-hot encoding for State
            crop_onehot = np.zeros(len(df['CROP'].unique()))    # One-hot encoding for Crop

            state_idx = list(df['STATE'].unique()).index(State)
            crop_idx = list(df['CROP'].unique()).index(Crop)

            state_onehot[state_idx] = 1
            crop_onehot[crop_idx] = 1

            # Prepare the query with numerical features and one-hot encoded categorical features
            query = np.concatenate([[N_SOIL, P_SOIL, Temp, Humidity, ph, Rainfall], state_onehot, crop_onehot])

            # If the total number of columns exceeds 48, drop random columns from one-hot encodings
            if query.shape[0] > 49:
                total_to_drop = query.shape[0] - 49  # Number of excess columns
                onehot_indices = list(range(6, query.shape[0]))  # Indices of one-hot encoded columns (after numerical inputs)
                random_indices_to_drop = random.sample(onehot_indices, total_to_drop)  # Randomly select indices to drop
                query = np.delete(query, random_indices_to_drop)  # Drop selected indices

            # Check if the total number of features matches 48
            if query.shape[0] == 49:
                query = query.reshape(1, -1)  # Reshape to match the input expected by the model

                # Predict price
                predicted_price = pipe.predict(query)[0]

                # Check if prediction is a valid number
                if np.isfinite(predicted_price):
                    predicted_price = int((predicted_price))  # Assuming log transformation on prices
                    pr1 = int(predicted_price )  # Add a 2.5% margin if needed
                    st.success(f"The predicted price of this crop is {pr1:,}")
                else:
                    st.error("Prediction resulted in an invalid value (infinity or NaN).")
            else:
                st.error(f"Feature length mismatch! Expected 48, got {query.shape[0]}.")

    except ZeroDivisionError:
        st.error("Screen size must be greater than zero to calculate PPI.")
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
