import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Prediction App

This app predicts the **Advertising** sale!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 0.7, 296.4, 147.0)
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 23.3)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114.0, 30.6)
    
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("AdvertisingSalesWani.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
