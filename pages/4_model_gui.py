import streamlit as st
import pandas as pd
import plotly.express as px


st.title(' :bar_chart: Model GUI')


classifier_name = st.sidebar.selectbox('Select Classifier', ('Decision Tree', 'Random Forest'))

