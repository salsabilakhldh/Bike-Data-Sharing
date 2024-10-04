import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_hour = pd.read_csv('https://raw.githubusercontent.com/salsabilakhldh/Bike-Data-Sharing/refs/heads/main/hour.csv')
data_day = pd.read_csv('https://raw.githubusercontent.com/salsabilakhldh/Bike-Data-Sharing/refs/heads/main/day.csv')

# Title of the Streamlit App
st.title('Bike Sharing Data Analysis')

# Sidebar for dataset selection
option = st.sidebar.selectbox('Choose dataset', ('Hour', 'Day'))

# Load the selected dataset
if option == 'Hour':
    st.write('You selected Hourly dataset')
    data = data_hour
else:
    st.write('You selected Daily dataset')
    data = data_day

# Show raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Analysis
st.subheader('Descriptive Statistics')
st.write(data.describe())

# Correlation analysis between temperature and total users
st.subheader('Correlation Analysis')
correlation = data['temp'].corr(data['cnt'])
st.write(f'Correlation between temperature and total bike users: {correlation}')

# Visualizations
st.subheader('Visualizations')

# Scatterplot: Temperature vs Total Users
st.write('Scatterplot: Temperature vs Total Users')
fig, ax = plt.subplots()
sns.scatterplot(x=data['temp'], y=data['cnt'], ax=ax)
ax.set_title('Temperature vs Total Users')
ax.set_xlabel('Temperature')
ax.set_ylabel('Total Users')
st.pyplot(fig)

# Scatterplot: Hour (for Hourly dataset) or Month (for Daily dataset) vs Total Users
if option == 'Hour':
    st.write('Scatterplot: Hour vs Total Users')
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['hr'], y=data['cnt'], ax=ax)
    ax.set_title('Hour vs Total Users')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Total Users')
    st.pyplot(fig)
else:
    st.write('Scatterplot: Month vs Total Users')
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['mnth'], y=data['cnt'], ax=ax)
    ax.set_title('Month vs Total Users')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Users')
    st.pyplot(fig)

# Cluster Analysis
st.subheader('Cluster Analysis')
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select features for clustering
if option == 'Hour':
    features = data[['season', 'hr', 'temp', 'hum', 'windspeed', 'cnt']]
else:
    features = data[['season', 'mnth', 'temp', 'hum', 'windspeed', 'cnt']]

# Normalize data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(features_scaled)

# Plot clustering result
st.write('Clustering Result Visualization')
if option == 'Hour':
    st.write('Clustering based on Hour and Total Users')
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['hr'], y=data['cnt'], hue=data['cluster'], palette='viridis', ax=ax)
    ax.set_title('Clustering: Hour vs Total Users')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Total Users')
    st.pyplot(fig)
else:
    st.write('Clustering based on Month and Total Users')
    fig, ax = plt.subplots()
    sns.scatterplot(x=data['mnth'], y=data['cnt'], hue=data['cluster'], palette='viridis', ax=ax)
    ax.set_title('Clustering: Month vs Total Users')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Users')
    st.pyplot(fig)
