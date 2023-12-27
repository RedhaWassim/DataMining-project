import streamlit as st
import pandas as pd
import plotly.express as px


st.title(' :bar_chart: Second Data Dashboard')

# Load data
df = pd.read_csv(r"/home/redha/Documents/projects/NLP/streamlit test/streamlit_test/data/second_df.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)   

# Data preprocessing
# Convert Start date to datetime
df['Start date'] = pd.to_datetime(df['Start date'])
#convert end date to datetime
df['end date'] = pd.to_datetime(df['end date'])
df['Year'] = df['Start date'].dt.year
df['Month'] = df['Start date'].dt.month_name()
df['Week'] = df['Start date'].dt.isocalendar().week

# Sidebar for user input
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Select a year for scatter', df['Year'].unique())
selected_zone = st.sidebar.selectbox('Select a zone for scatter', df['zcta'].unique())

# Filter data based on user input
filtered_data = df[(df['zcta'] == selected_zone) & (df['Year'] == selected_year)]

# Layout
st.subheader('Dataset Overview')
st.dataframe(filtered_data.head())

# Scatter Plot
st.subheader('Scatter Plot')
fig_scatter = px.scatter(filtered_data, x='case rate', y='test rate', color='positivity rate', title='Case Rate vs. Test Rate')
st.plotly_chart(fig_scatter)

# Tree Map/Bar Chart
st.subheader('Confirmed Cases and Positive Tests by Zone')
fig_tree = px.treemap(df, path=['zcta', 'Year'], values='case count', title='Confirmed Cases by Zone and Year')
st.plotly_chart(fig_tree)

# Corrected Stacked Bar Chart showing COVID Positive Cases Distribution by Zone and Year
st.subheader('COVID Positive Cases Distribution by Zone and Year')

# Aggregating data by 'zcta' and 'Year'
agg_data_by_zone_year = df.groupby(['zcta', 'Year'], as_index=False)['positive tests'].sum()

# Plot the treemap with aggregated data
fig_treemap = px.treemap(
    agg_data_by_zone_year,
    path=['zcta', 'Year'],  # Defines the hierarchical levels
    values='positive tests',
    color='positive tests',
    color_continuous_scale='RdYlGn',
    title='Positive Cases by Zone and Year'
)

st.plotly_chart(fig_treemap)


# Population vs. Test Count with distinct colors for each zone
st.subheader('Population vs. Test Count')
fig_bubble = px.scatter(
    df, 
    x='population', 
    y='test count', 
    size='case count', 
    color='zcta',  # This should automatically provide a unique color for each zcta value
    title='Population vs. Test Count by Zone',
    labels={"zcta": "Zone"}  # Rename legend title
)

# Customize the legend to show one color per 'zcta' and remove the color scale
fig_bubble.update_layout(
    coloraxis_showscale=False,  # Hide the color scale
    legend_title_text='Zone'
)

st.plotly_chart(fig_bubble)


# Top 5 Impacted Zones
st.subheader('Top 5 Impacted Zones')
top_zones = df.groupby('zcta')['case count'].sum().nlargest(5)
st.bar_chart(top_zones)







# User input for zone selection
selected_zone = st.sidebar.selectbox('Select a zone for time series analysis', df['zcta'].unique())

# Filter the dataframe for the selected zone
zone_data = df[df['zcta'] == selected_zone]

# Prepare data for weekly, monthly, and annual trends
zone_data['Week'] = zone_data['Start date'].dt.isocalendar().week
zone_data['Month'] = zone_data['Start date'].dt.strftime('%Y-%m')
zone_data['Year'] = zone_data['Start date'].dt.year

# Group the data for different time scales
weekly_data = zone_data.groupby('Week').agg({'test count':'sum', 'positive tests':'sum', 'case count':'sum'}).reset_index()
monthly_data = zone_data.groupby('Month').agg({'test count':'sum', 'positive tests':'sum', 'case count':'sum'}).reset_index()
annual_data = zone_data.groupby('Year').agg({'test count':'sum', 'positive tests':'sum', 'case count':'sum'}).reset_index()

# Define a function to create line charts
def create_line_chart(data, x_axis, title):
    fig = px.line(data, x=x_axis, y=['test count', 'positive tests', 'case count'], markers=True, title=title)
    return fig

# Plot weekly, monthly, and annual trends
st.subheader('Weekly Trend for Zone ' + str(selected_zone))
st.plotly_chart(create_line_chart(weekly_data, 'Week', 'Weekly Trend'))

st.subheader('Monthly Trend for Zone ' + str(selected_zone))
st.plotly_chart(create_line_chart(monthly_data, 'Month', 'Monthly Trend'))

st.subheader('Annual Trend for Zone ' + str(selected_zone))
st.plotly_chart(create_line_chart(annual_data, 'Year', 'Annual Trend'))
