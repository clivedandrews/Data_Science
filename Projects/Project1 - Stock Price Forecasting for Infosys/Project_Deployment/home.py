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

# This is the function that fetches the Company's ticker
def get_ticker(name):
    company = fn.Ticker(name)
    return company

# Function to replace outliers
def replace_outlier(df_in, col_name):
    Q1 = df_in[col_name].quantile(0.25)
    Q3 = df_in[col_name].quantile(0.75)
    IQR = Q3 - Q1 # Interquartile range
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR
    median_val = df_in[col_name].median()
    df_out = df_in.copy()
    for i, j in df_in.iterrows():
        if (j[col_name] > upper):
            df_out.at[i, col_name] = upper
        elif (j[col_name] < lower):
            df_out.at[i, col_name] = lower
        else:
            df_out.at[i, col_name] = df_out.at[i, col_name]
    return df_out
    
# Function to check for outliers
def check_outliers(df_in, col_name):
    Q1 = df_in[col_name].quantile(0.25)
    Q3 = df_in[col_name].quantile(0.75)
    IQR = Q3 - Q1 # Interquartile range
    upper = Q3 + 1.5*IQR
    lower = Q1 - 1.5*IQR
    outliers = False
    for i, j in df_in.iterrows():
        if (j[col_name] > upper):
            outliers = True
        elif (j[col_name] < lower):
            outliers = True
    return outliers


def extract_prediction_details():
    # Importing the model
    import pickle
    lstm_model = pickle.load(open('lstm_model.pkl','rb'))
    lstm_model.compile(loss='mean_squared_error',optimizer='adam')

    if 'Infosys_Updated' not in st.session_state:

        # Importing the data from the file 'arima_df_output.csv'
        lstm_df = pd.read_csv('lstm_df_output.csv')
        inf_data = st.session_state['Infosys_latest']


        # Removing the column 'Unnamed: 0'
        lstm_df = lstm_df.drop(['Unnamed: 0'], axis=1)
        lstm_df = lstm_df[["DATE", "STOCK_PRICE"]]
        lstm_df['DATE'] = pd.to_datetime(lstm_df['DATE'], format = '%d-%m-%Y')
        all_dates = pd.to_datetime(lstm_df['DATE'], format = '%d-%m-%Y').dt.strftime('%Y-%m-%d').to_list()
        train_size = int(len(lstm_df)*0.95)
        test_size = len(lstm_df) - train_size

        # This block will add data for the latest 5 days data not included in the file 'lstm_df_output.csv'
        # If older than 5 days is missing, it needs to be extracted as part of the file 'lstm_df_output.csv' first
        from datetime import datetime
        for index, row in inf_data.iterrows():
            ldate = pd.to_datetime(index, format = '%Y-%m-%d %H:%M:%S%f%z').strftime('%Y-%m-%d') # Getting string in the format yyyy-mm-dd
            tdate = pd.to_datetime(index, format = '%Y-%m-%d %H:%M:%S%f%z').strftime('%d-%m-%Y') # Getting string in the format yyyy-mm-dd
            if ldate not in all_dates:
                new_index = len(lstm_df.index)+1
                stock_val = row["Close"]
                all_dates.append(ldate)
                lstm_df.loc[new_index, 'DATE'] = tdate
                lstm_df.loc[new_index, 'STOCK_PRICE'] = stock_val
        
        # Marking each record as Train or Test based on train_size, test_size
        for index, row in lstm_df.iterrows():
            if (index <= train_size):
                lstm_df.at[index, 'Indicator'] = 'train'
            elif (index > train_size):
                lstm_df.at[index, 'Indicator'] = 'test'

        st.session_state['Infosys_Updated'] = lstm_df


    else:
        lstm_df = st.session_state['Infosys_Updated']

        sdata_sc = pd.DataFrame()
        sdata_sc[['STOCK_PRICE']] = scaler.fit_transform(lstm_df[['STOCK_PRICE']])
        ds_scaled = sdata_sc[['STOCK_PRICE']].values

        get_100_days = sdata_sc.STOCK_PRICE.tail(100).values
        fut_inp = get_100_days
        tmp_inp = list(fut_inp)

        # Predicting next duration (configurable) days price using the current data
        # It will predict in sliding window manner (algorithm) with stride 1

        duration = 10
        lst_output=[]
        n_steps=100
        i=0
        while(i<duration):
            
            if(len(tmp_inp)>100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp=fut_inp.reshape(1,-1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                yhat = lstm_model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                tmp_inp = tmp_inp[1:]
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                fut_inp = fut_inp.reshape((1, n_steps,1))
                yhat = lstm_model.predict(fut_inp, verbose=0)
                tmp_inp.extend(yhat[0].tolist())
                lst_output.extend(yhat.tolist())
                i=i+1

        forecast_list = scaler.inverse_transform(lst_output).reshape(duration)

        import datetime
        final_stock_data = lstm_df.copy()
        final_stock_data['DATE'] = pd.to_datetime(final_stock_data['DATE'], format = '%Y-%m-%d')

        new_forecast = pd.DataFrame(columns=final_stock_data.columns)
        new_forecast['STOCK_PRICE'] = forecast_list
        new_forecast['Indicator'] = 'forecasted'

        # Getting the last row date from 'DATE' column, which can be used to get the start_date and end_date
        last_row_date = final_stock_data.DATE.iloc[-1] 
        last_row_date = pd.to_datetime(last_row_date, format = '%Y-%m-%d') # Last date of the data frame in formatted
        start_date = last_row_date + datetime.timedelta(days=1)

        # Adding duration number of days
        end_date = start_date + datetime.timedelta(days=duration)

        # Adding the new dates to the dataframe new_forecast
        new_forecast['DATE']= pd.DataFrame(pd.date_range(start=start_date, end=end_date), columns=['DATE'])
        frames = [lstm_df, new_forecast]
        final_stock_data = pd.concat(frames)

        #Creating final data for plotting
        ds_new = ds_scaled.tolist()
        final_graph = scaler.inverse_transform(ds_new).tolist()

        # Resetting the index to Date
        final_stock_data = final_stock_data.reset_index(drop=True)

        stock_data = final_stock_data.copy()
        stock_data['DATE'] = pd.to_datetime(stock_data['DATE'],format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
        stock_data['DATE'] = pd.to_datetime(stock_data['DATE'], format='%Y-%m-%d')

        # Adding MONTH, YEAR, YEAR-MONTH Columns
        stock_data["MONTH"] = stock_data['DATE'].dt.strftime("%b") # month extraction
        stock_data["YEAR"] = stock_data['DATE'].dt.strftime("%Y") # year extraction
        stock_data["YEAR-MONTH"] = stock_data['DATE'].dt.strftime("%Y-%m")
        st.session_state['Inter_DataSet'] = stock_data

        return final_stock_data

      

def main():

    st.title("Infosys Stock Price Forecasting")
    infs = get_ticker("INFY.BO")
    st.header('Simple Data Science Web App')

    infs = get_ticker("INFY.BO")
    
    # Fetching data from dataframe
    #infosys_df = fn.download("INFY.BO", start="2023-12-01", end="2023-12-11")
    
    # Fetching historical data by valid periods (1D, 5D, 3M, ...etc)
    # inf_data = infs.history(period="5d")
    inf_5d_data = infs.history(period="5d")
    inf_full_data = infs.history(period="Max")
    st.session_state['Infosys_latest'] = inf_full_data
    
    # Markdown
    st.write(""" ### INFOSYS """)
    
    # Detailed summary about Infosys
    st.write(infs.info['longBusinessSummary'])
    
    # Dataframe
    #st.write(infosys_df)
    st.write(inf_5d_data)

    final_stock_data = extract_prediction_details()
    st.session_state['Final_DataSet'] = final_stock_data
    
    # 5 Days Chart
    fig = plt.figure(figsize = (20, 6))
    sns.lineplot(x=pd.to_datetime(inf_5d_data.index, format = '%Y-%m-%d %H:%M:%S%f%z').strftime('%Y-%m-%d'), y='Close', palette=['#009f6b','#FF0000'], data=inf_5d_data)
    plt.xticks(rotation = 0,fontsize = 25, fontfamily = "Times New Roman")
    plt.yticks(rotation = 0,fontsize = 25, fontfamily = "Times New Roman")
    plt.xlabel("Date", weight='bold', fontfamily = "Times New Roman", size = '25')
    plt.ylabel("Close", weight='bold', fontfamily = "Times New Roman", size = '25')
    plt.title("STOCK PRICE LINE CHART\n\n", verticalalignment="center", weight='bold', fontfamily = "Times New Roman", size = '30')
    st.pyplot(fig)

    


if __name__ == '__main__':
    main()
        

        
        
        
        
        
        
        