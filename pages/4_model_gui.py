import streamlit as st
import pandas as pd
import plotly.express as px
from soil_fertility.utils import load_object, retreive_base_path
from soil_fertility.pipeline.predict_pipeline import InputData, PredictPipeline
from soil_fertility.logger import logging

st.title(' :bar_chart: Model GUI')

base_path = retreive_base_path()

classifier_name = st.sidebar.selectbox('Select Classifier', ('Decision Tree', 'Random Forest'))
classifier_type = st.sidebar.radio('Select Classifier', ('normal', 'grid search'))

classifier_name = classifier_name.lower().replace(' ', '_')
if classifier_type == 'grid search':
    classifier_type = '_'+classifier_type.lower().replace(' ', '')
else:
    classifier_type = '' 
model_name = f'{classifier_name}{classifier_type}.pkl'

col1, col2, col3, col4 = st.columns(4)

with col1:
    N = st.number_input('Enter a value for N' )
    P = st.number_input('Enter a value for P' )
    K = st.number_input('Enter a value for K' )

with col2:
    pH = st.number_input('Enter a value for pH')
    EC = st.number_input('Enter a value for EC')
    OC = st.number_input('Enter a value for OC')

with col3:
    S = st.number_input('Enter a value for S')
    Zn = st.number_input('Enter a value for Zn')
    Fe = st.number_input('Enter a value for Fe')
with col4:
    Cu = st.number_input('Enter a value for Cu')
    Mn = st.number_input('Enter a value for Mn')
    B = st.number_input('Enter a value for B')

OM = st.number_input('Enter a value for OM')

def predict(model_name):
    data = InputData(
        N=N,
        P=P,
        K=K,
        pH=pH,
        EC=EC,
        OC=OC,
        S=S,
        Zn=Zn,
        Fe=Fe,
        Cu=Cu,
        Mn=Mn,
        B=B,
        OM=OM
    )
    df=data.get_data_as_df()
    pipeline = PredictPipeline()
    prediction = pipeline.predict(df, model_name)

    st.write(f'Prediction : {prediction}')



st.button('Predict',on_click=predict(model_name))




