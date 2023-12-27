import streamlit as st
import pandas as pd
import plotly.express as px


st.title(" :bar_chart: First Data Dashboard")

# Load data
df = pd.read_csv(
    r"/home/redha/Documents/projects/NLP/streamlit test/streamlit_test/data/first_df.csv"
)

# Sidebar for user input
st.sidebar.header("User Input Features")
selected_histogram = st.sidebar.selectbox(
    "Select a variable for histogram", df.columns[:-1]
)
selected_boxplot = st.sidebar.selectbox(
    "Select a variable for boxplot", df.columns[:-1]
)
selected_x_scatter = st.sidebar.selectbox(
    "Select X variable for scatter plot", df.columns[:-1], index=0
)
selected_y_scatter = st.sidebar.selectbox(
    "Select Y variable for scatter plot", df.columns[:-1], index=1
)
selected_bar = st.sidebar.selectbox(
    "Select a metric for comparative analysis", df.columns[:-1]
)

# Layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.write("Dataset Overview:")
    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.dataframe(df.head())

with col2:
    st.write("Descriptive Statistics:")
    st.write(df.describe().T)

with col3:
    st.subheader("Correlation Heatmap")
    fig_corr = px.imshow(
        df.corr(), x=df.columns, y=df.columns, color_continuous_scale="Viridis"
    )
    fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Histogram")
    fig_hist = px.histogram(df, x=selected_histogram, marginal="box", nbins=30)
    fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Box Plot")
    fig_box = px.box(df, y=selected_boxplot)
    fig_box.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_box, use_container_width=True)

st.subheader("Scatter Plot")
fig_scatter = px.scatter(
    df, x=selected_x_scatter, y=selected_y_scatter, color="Fertility"
)
fig_scatter.update_layout(margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig_scatter, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    # Comparative Analysis by Fertility Level
    st.subheader("Comparative Analysis by Fertility Level")
    fig_comp = px.bar(
        df.groupby("Fertility")[selected_bar].mean().reset_index(),
        x="Fertility",
        y=selected_bar,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with col2:
    # Pie Chart for Categorical Data
    # Assuming 'Fertility' is a categorical variable
    st.subheader("Distribution of Fertility Levels")
    fig_pie = px.pie(df, names="Fertility", title="Fertility Levels Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
