import streamlit as st
import pandas as pd
import plotly.express as px


st.title(" :bar_chart: Third Data Dashboard")

# Load data
df = pd.read_excel(
    r"/home/redha/Documents/projects/NLP/streamlit test/streamlit_test/data/Dataset3.xlsx"
)

st.sidebar.header("User Input Features")
selected_histogram = st.sidebar.selectbox("Histogram Variable", df.columns[:-1])
selected_pie = st.sidebar.selectbox(
    "Pie Chart Category", ["Soil", "Crop", "Fertilizer"]
)
selected_scatter_x = st.sidebar.selectbox(
    "X-axis for Scatter Plot", df.columns[:-3], index=0
)
selected_scatter_y = st.sidebar.selectbox(
    "Y-axis for Scatter Plot", df.columns[:-3], index=1
)
col1, col2 = st.columns(2)

with col1:
    st.write("Dataset Overview:")
    st.dataframe(df)

with col2:
    st.write("Descriptive Statistics:")
    st.write(df.describe().T)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Histogram")
    fig_hist = px.histogram(df, x=selected_histogram)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Pie Chart")
    fig_pie = px.pie(df, names=selected_pie)
    st.plotly_chart(fig_pie, use_container_width=True)

col1, col2 = st.columns(2)

st.subheader("Scatter Plot")
fig_scatter = px.scatter(df, x=selected_scatter_x, y=selected_scatter_y)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Scatter Plot Matrix")
fig_scatter_matrix = px.scatter_matrix(
    df, dimensions=["Temperature", "Humidity", "Rainfall"], color="Crop"
)
st.plotly_chart(fig_scatter_matrix, use_container_width=True)


st.subheader("Crop Frequency Bar Plot")

# Count the frequency of each crop and reset the index
crop_count = df["Crop"].value_counts().reset_index()

# Rename the columns for clarity
crop_count.columns = ["Crop", "Frequency"]

# Create the bar plot
fig_crop_count = px.bar(crop_count, x="Crop", y="Frequency")

# Display the plot
st.plotly_chart(fig_crop_count, use_container_width=True)

st.subheader("Temperature and Rainfall Distribution by Soil Type")
col1, col2 = st.columns(2)
with col1:
    fig_temp_soil = px.box(df, x="Soil", y="Temperature")
    st.plotly_chart(fig_temp_soil, use_container_width=True)
with col2:
    fig_rain_soil = px.box(df, x="Soil", y="Rainfall")
    st.plotly_chart(fig_rain_soil, use_container_width=True)


# Add a select box to the sidebar for the user to choose the treemap category
st.sidebar.header("Treemap Options")
treemap_option = st.sidebar.selectbox(
    "Select the category for treemap:", options=["Crop", "Soil"]
)

st.subheader(f"Treemap of Fertilizer Usage by {treemap_option}")

# Generate the treemap based on the selected option
fig_treemap = px.treemap(df, path=[treemap_option, "Fertilizer"], values="Rainfall")
st.plotly_chart(fig_treemap, use_container_width=True)
