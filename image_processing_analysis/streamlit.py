import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient

st.set_page_config(
    page_title="Image Meta-Tag Charts",
    layout='centered',
    initial_sidebar_state='auto',
    menu_items={
        'About': "This is a header. This is an *extremely* cool app!"
    }
)

st.set_option('deprecation.showPyplotGlobalUse', False)

D = pd.read_csv('50kdataset.csv')
df = D

# --------------------- Streamlit ------------------------#

# Set page header
st.header('Data Visualization Dashboard')

chart_type = st.selectbox("Select a chart type", ["Pie chart", "Bar chart", "Histogram"])

# Display the dataframe
st.write(df)


def barplot_data(data, variables):
    for variable in variables:
        # Barplot
        plt.figure(figsize=(15, 10))
        sns.barplot(x=variable, y='count', data=data.groupby(variable).size().reset_index(name='count'))
        plt.title(f'Barplot of {variable}')
        st.pyplot()


def hist_data(data, variables):       
    for variable in variables:
        # Histogram
        plt.figure(figsize=(15, 10))
        sns.histplot(data[variable], kde=True)
        plt.title(f'Histogram of {variable}')
        st.pyplot()
        

def pie_data(data, variables):
    for variable in variables: 
        # Pie Chart
        plt.style.use('default')
        plt.figure(figsize=(15, 10))
        plt.pie(data.groupby(variable).size(), labels=(data[variable].unique()), explode=(0.1, 0.1), autopct='%1.2f%%', shadow=True)
        plt.legend()
        plt.title(f'Pie Chart of {variable}')
        print('\n bad - 10795.70 : good - 38432.29 \n duplicate - 2717.3')
        st.pyplot()

data = df

# Specify variables to plot
variables1 = ['duplicate', 'image_quality_flag']

# Specify variables to plot
variables2 = ['height','width','size','ppi','red','blue','green','megapixels','laplacian_variance_blur','fourier_transform_blur','gradient_magnitude_blur']

# Generate plots
if chart_type == "Pie chart":
    pie_data(data, variables1)

elif chart_type == "Bar chart":
    barplot_data(data, variables1)

elif chart_type == "Histrogram":
    hist_data(data, variables2)