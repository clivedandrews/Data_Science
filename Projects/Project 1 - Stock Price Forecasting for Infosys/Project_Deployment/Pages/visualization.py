# https://github.com/pavanwanjari/StockPrice_ModelDeployment_Streamlit/blob/main/Model_Deployment_Streamlit/app.py
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:30:25 2023

@author: excel
"""

import yfinance as fn
import streamlit as st
import datetime
import statsmodels.api as sm

# Importing Libraries
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
# from pmdarima.arima import auto_arima
from pandas import DataFrame
from pandas import Grouper
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, accuracy_score
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from collections import Counter
import plotly.express as px
from plotly.offline import plot as off
import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, \
                                    train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()






def main():
    
    # start_date = st.date_input("Enter the Date in this format yyyy-mm-dd")
    st.title("Infosys Stock Visualization")

    # Getting the latest Infosys Stock Details stored in the dataframe extracted in home page
    stock_symbol = "Infosys Limited (INFY.BO)"
    duration = 10

    # Code to extract data from another page based on data saved in a session.
    # Check if you've already initialized the data
    if 'Infosys_Updated' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
        exit()
    else:
        # Getting the latest Infosys Stock Details stored in the dataframe extracted in home page
        stock_data = st.session_state['Infosys_Updated']

    if 'Inter_DataSet' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
        exit()
    else:
        # Getting the Intermediate Infosys Stock Details stored in the dataframe extracted in home page
        inter_stock_data = st.session_state['Inter_DataSet']


    if 'Final_DataSet' not in st.session_state:
        # Get the data if you haven't
        st.error('Some forecasting data is not extracted, please go back to \'home\' and then to \'stock prediction\' page before seeing the visualization.')
        exit()
    else:
        # Getting the final Infosys Stock Details stored in the dataframe extracted in home page
        final_stock_data = st.session_state['Final_DataSet']
        
    
    # Creating a visual of the Train, Test and Forecast data for Closing Stock Price
    st.header('Test, Train and Forecast Data for Closing Stock Price')
    st.write(final_stock_data)
    fig = plt.figure(figsize = (20, 6))
    sns.lineplot(x='DATE', y='STOCK_PRICE', hue='Indicator', palette=['#0541ff','#009f6b','#FF0000'], data=final_stock_data)
    plt.xticks(rotation = 0,fontsize = 25, fontfamily = "Times New Roman")
    plt.yticks(rotation = 0,fontsize = 25, fontfamily = "Times New Roman")
    plt.xlabel("DATE", weight='bold', fontfamily = "Times New Roman", size = '25')
    plt.ylabel("STOCK COUNT", weight='bold', fontfamily = "Times New Roman", size = '25')
    plt.title(f"{stock_symbol} prediction of next {duration} Days", verticalalignment="center", weight='bold', fontfamily = "Times New Roman", size = '30')
    plt.legend()
    st.pyplot(fig)

    # Creating a visual of the Stock Price Heatmap
    # st.header('Stock Price Heatmap')
    fig_hm = plt.figure(figsize=(20,10))
    hm_month = pd.pivot_table(data=inter_stock_data,values="STOCK_PRICE",index="YEAR",columns="MONTH",aggfunc="mean",fill_value=0)
    sns.heatmap(hm_month, annot=True, linewidths =.5, fmt ='.2f',cmap="YlGnBu")
    plt.xticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.yticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.xlabel("MONTH", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.ylabel("YEAR", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.title("STOCK COUNT HEATMAP\n\n", verticalalignment="center", weight='bold', fontfamily = "Times New Roman", size = '30')
    st.pyplot(fig_hm)

    # Creating a visual of Yearly Trend
    # st.header('Stock Yearly Trend')
    fig_yt = plt.figure(figsize=(20,5))
    sns.lineplot(x="YEAR",y="STOCK_PRICE",data=inter_stock_data)
    plt.xticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.yticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.xlabel("YEARS", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.ylabel("STOCK COUNT", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.title("YEARLY TREND\n\n", verticalalignment="center", weight='bold', fontfamily = "Times New Roman", size = '30')
    st.pyplot(fig_yt)

    # Boxplot for Year Data
    fig_yo = plt.figure(figsize=(20,10))
    sns.boxplot(x="YEAR",y="STOCK_PRICE",data=inter_stock_data, palette="rainbow")
    plt.xticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.yticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.xlabel("YEARS", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.ylabel("STOCK_PRICE COUNT", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.title("YEARLY OUTLIERS", weight='bold', fontfamily = "Times New Roman", size = '30')
    st.pyplot(fig_yo)

    # Creating a visual of Monthly Trend
    # st.header('Stock Monthly Trend')
    fig_mt = plt.figure(figsize=(20,5))
    sns.lineplot(x="MONTH",y="STOCK_PRICE",data=inter_stock_data)
    plt.xticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.yticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.xlabel("MONTHS", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.ylabel("STOCK COUNT", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.title("MONTHLY TREND\n\n", verticalalignment="center", weight='bold', fontfamily = "Times New Roman", size = '30')
    st.pyplot(fig_mt)

    # Boxplot for Monthly Data
    fig_mo = plt.figure(figsize=(20,10))
    sns.boxplot(x="MONTH",y="STOCK_PRICE",data=inter_stock_data, palette="rainbow")
    plt.xticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.yticks(rotation = 0,fontsize = 20, fontfamily = "Times New Roman")
    plt.xlabel("MONTHS", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.ylabel("STOCK_PRICE COUNT", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.title("MONTHLY OUTLIERS", weight='bold', fontfamily = "Times New Roman", size = '30')
    st.pyplot(fig_mo)

    # Creating a Histogram and Density Plot
    # st.header('Histogram and Density Plot')
    fig_hdp = plt.figure(figsize=(20,10))
    plot = sns.histplot(inter_stock_data, x='STOCK_PRICE', color='#4c9a47', edgecolor='#4c9a47', linewidth=2, bins=10, kde=True)
    plt.setp(plot.get_xticklabels(), rotation=0, fontfamily = "Times New Roman", size = '20')
    plt.setp(plot.get_yticklabels(), rotation=0, fontfamily = "Times New Roman", size = '20')
    plt.xlabel('STOCK_PRICE', weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.ylabel("COUNT", weight='bold', fontfamily = "Times New Roman", size = '20')
    plt.title('HISTOGRAM AND DENSITY PLOT', weight='bold', fontfamily = "Times New Roman", size = '30')
    st.pyplot(fig_hdp)



        

if __name__ == '__main__':
    main()
