import streamlit as st
import pickle
import pandas as pd

# Load the model and encoders
with open('model_penguin_65130700333.pkl', 'rb') as file:
   model, species_encoder, island_encoder, sex_encoder = pickle.load(file)

# Create a title
st.title("Penguin Species Prediction")

# Get user input
island = st.selectbox("Island", ["Torgersen", "Biscoe", "Dream"])
culmen_length = st.number_input("Culmen Length (mm)", min_value=0.0, max_value=100.0, value=37.0)
culmen_depth = st.number_input("Culmen Depth (mm)", min_value=0.0, max_value=100.0, value=19.3)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0.0, max_value=300.0, value=192.3)
body_mass = st.number_input("Body Mass (g)", min_value=0.0, max_value=6000.0, value=3000.0)
sex = st.selectbox("Sex", ["MALE", "FEMALE"])

# Prepare input data
x_new = pd.DataFrame({'island': [island],
                       'culmen_length_mm': [culmen_length],
                       'culmen_depth_mm': [culmen_depth],
                       'flipper_length_mm': [flipper_length],
                       'body_mass_g': [body_mass],
                     'sex': [sex]})

# Predict the species
#if st.button("Predict"):
    #y_pred_new = model.predict(x_new)
    #predicted_species = species_encoder.inverse_transform(y_pred_new)[0]
    #st.write("Predicted Species:", predicted_species)
