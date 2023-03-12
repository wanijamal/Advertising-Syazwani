import streamlit as st
import pandas as pd
import pickle

st.write("""
# Advertising Prediction App

This app predicts the **Advertising** sale!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('TV', 4.3, 7.9, 5.4)
    Radio = st.sidebar.slider('Radio', 2.0, 4.4, 3.4)
    Newspaper = st.sidebar.slider('Newspaper', 1.0, 6.9, 1.3)
    Sales = st.sidebar.slider('Sales', 0.1, 2.5, 0.2)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper,
            'Sales': Sales}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("AdvertisingSalesWani.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write(prediction)
