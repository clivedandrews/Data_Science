"""
Created on Sat Jan 20 10:42:05 2024

@author: Clive Dominic Andrews
"""

# import the libraries
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import random
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from streamlit import session_state as ss
palette_color = sns.color_palette('rainbow')

def main():
    
    st.title("LIVER DISEASE DATA VISUALIZATION")

    # Code to extract data from another page based on data saved in a session.
    # Check if you've already initialized the data
    if 'LDC_LED' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
    else:
        # Getting the latest Liver Disease Dataframe extracted in home page
        ldcdata = st.session_state['LDC_LED']
    
    if 'LDC_NORM' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
    else:
        # Getting the latest Liver Disease Normalized Dataframe from in home page
        ldc_norm = st.session_state['LDC_NORM']

    if 'LDC_INIT' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
    else:
        # Getting the latest Liver Disease Dataframe extracted in home page
        ldc_data = st.session_state['LDC_INIT']
        
    
    if 'LDC_STD' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
    else:
        # Getting the latest Liver Disease Standardized Dataframe from in home page
        ldc_ss = st.session_state['LDC_STD']

    # Segregating categorical Columns and numerical columns
    categorical_cols = []
    numeric_cols = []
    for columns in ldc_data.columns:
      if ((ldc_data[columns].dtypes == 'object') or (ldc_data[columns].dtypes == 'category')):
            categorical_cols.append(columns)
      elif ((ldc_data[columns].dtypes != 'object') and (ldc_data[columns].dtypes != 'category')):
            numeric_cols.append(columns)
    
    # Creating dataframes for both categorical data as well as numeric data
    ldc_cat_ldc_data = pd.DataFrame()
    ldc_num_ldc_data = pd.DataFrame()

    ldc_cat_ldc_data = ldc_data[categorical_cols]
    ldc_num_ldc_data = ldc_data[numeric_cols]

    # Creating Histograms and Box Plots for Numeric Columns
    st.header('Histogram Columns - Histogram and Box-Plot')
    def plot_data(data,feature, block=None):
        fig_num_hist_box = plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        plot = sns.histplot(data, x=feature, color=color, edgecolor=color, linewidth=2, bins=10, kde=True)
        plt.xticks(rotation=0, fontfamily = "Times New Roman", size = '11')
        plt.yticks(rotation=0, fontfamily = "Times New Roman", size = '11')
        plt.xlabel(feature, rotation=0, fontfamily = "Times New Roman", size = '11', weight='bold')
        plt.ylabel("COUNT", rotation=90, fontfamily = "Times New Roman", size = '11', weight='bold')
        plt.show(block=False)
        

        plt.subplot(1,2,2)
        sns.boxplot(x=data[feature].value_counts(), color=color)
        plt.setp(plot.get_xticklabels(), rotation=0, fontfamily = "Times New Roman", size = '11')
        plt.setp(plot.get_yticklabels(), rotation=0, fontfamily = "Times New Roman", size = '11')
        plt.xlabel("COUNT", rotation=0, fontfamily = "Times New Roman", size = '11', weight='bold')
        plt.suptitle(feature, weight='bold', fontfamily = "Times New Roman", size = '15')
        plt.show(block=False)
        st.pyplot(fig_num_hist_box)

    for n_cols in ldc_num_ldc_data:
        colors = ['#4c9a47','#90c08c','#598eff','#636300','#74a3ff','#bebe00','#7ab375','#808000','#0541ff','#9e9e00','#ffc274','#de8f00','#64a65e','#2260ff','#ffb85c','#3e78ff']
        color = random.choice(colors)
        plot_data(ldc_num_ldc_data,n_cols)


    
    # Creating Histograms and Box Plots for Categorical Columns
    st.header('Categorical Columns - Histogram and Box-Plot')
    def plot_data(data,feature):
        fig_cat_hist_box = plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        plt.xticks(rotation=90, fontfamily = "Times New Roman", size = '11')
        plt.yticks(rotation=0, fontfamily = "Times New Roman", size = '11')
        plt.xlabel("COUNT", rotation=90, fontfamily = "Times New Roman", size = '11', weight='bold')
        plt.xlabel(feature, rotation=0, fontfamily = "Times New Roman", size = '11', weight='bold')
        plot = sns.histplot(data, x=feature, color=color, edgecolor=color, linewidth=2, bins=10, kde=True)
        plt.show(block=False)

        plt.subplot(1,2,2)
        sns.boxplot(x=data[feature].value_counts(), color=color)
        plt.xticks(rotation=0, fontfamily = "Times New Roman", size = '11')
        plt.xlabel("COUNT", rotation=0, fontfamily = "Times New Roman", size = '11', weight='bold')
        plt.suptitle(feature, weight='bold', fontfamily = "Times New Roman", size = '15')
        plt.show(block=False)
        st.pyplot(fig_cat_hist_box)

    for c_cols in ldc_cat_ldc_data:
        colors = ['#4c9a47','#90c08c','#598eff','#636300','#74a3ff','#bebe00','#7ab375','#808000','#0541ff','#9e9e00','#ffc274','#de8f00','#64a65e','#2260ff','#ffb85c','#3e78ff']
        color = random.choice(colors)
        plot_data(ldc_cat_ldc_data,c_cols)

    
    # Creating Histograms and Box Plots for Categorical Columns
    fig_category_hist_pie = plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    palette_color = sns.color_palette('bright')
    sns.countplot(x='CATEGORY', data=ldc_cat_ldc_data, palette = palette_color, order=ldc_cat_ldc_data['CATEGORY'].value_counts().index )
    plt.xticks(rotation=90, fontfamily = "Times New Roman", size = '11')
    plt.yticks(rotation=0, fontfamily = "Times New Roman", size = '11')
    plt.ylabel("COUNT", rotation=90, fontfamily = "Times New Roman", size = '11', weight='bold')
    plt.xlabel('CATEGORY', rotation=0, fontfamily = "Times New Roman", size = '11', weight='bold')
    plt.show(block=False)

    # Display state data in a form of Pie Graph
    plt.subplot(1,2,2)
    values = ldc_cat_ldc_data['CATEGORY'].value_counts().keys().tolist()
    counts = ldc_cat_ldc_data['CATEGORY'].value_counts().tolist()
    plt.pie(ldc_cat_ldc_data['CATEGORY'].value_counts(), labels=values, colors=palette_color, autopct='%.0f%%')
    plt.suptitle('CATEGORY', weight='bold', rotation=0, fontfamily = "Times New Roman", size = '15')
    plt.show(block=False)
    st.pyplot(fig_category_hist_pie)


    fig_sex_hist_pie = plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    palette_color = sns.color_palette('rainbow')
    sns.countplot(x='SEX', data=ldc_cat_ldc_data, palette = palette_color, order=ldc_cat_ldc_data['SEX'].value_counts().index )
    plt.xticks(rotation=0, fontfamily = "Times New Roman", size = '11')
    plt.yticks(rotation=0, fontfamily = "Times New Roman", size = '11')
    plt.ylabel("COUNT", rotation=90, fontfamily = "Times New Roman", size = '11', weight='bold')
    plt.xlabel('CATEGORY', rotation=0, fontfamily = "Times New Roman", size = '11', weight='bold')

    # Display state data in a form of Pie Graph
    plt.subplot(1,2,2)
    values = ldc_cat_ldc_data['SEX'].value_counts().keys().tolist()
    counts = ldc_cat_ldc_data['SEX'].value_counts().tolist()
    plt.pie(ldc_cat_ldc_data['SEX'].value_counts(), labels=values, colors=palette_color, autopct='%.0f%%')
    plt.suptitle('SEX', weight='bold', rotation=0, fontfamily = "Times New Roman", size = '15')
    plt.show()
    st.pyplot(fig_sex_hist_pie)

    # Correlation Heatmap
    import plotly.express as px
    fig_hm, ax = plt.subplots(figsize=(9,3))
    st.header('Correlation Heatmap')
    sns.heatmap(ldcdata.corr(), annot=True, linewidths =1, fmt ='.2f',cmap="YlGnBu",ax=ax)
    plt.show()
    st.pyplot(fig_hm)

if __name__ == '__main__':
    main()
