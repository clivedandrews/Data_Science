"""
Created on Sat Jan 20 10:42:05 2024

@author: Clive Dominic Andrews
"""

# # import the libraries
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
# Clustering libraries
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
import scipy.cluster.hierarchy as sch # to build dendrogram and build the plotting
import warnings
warnings.filterwarnings('ignore')
scaler = MinMaxScaler()


  

def main():

    st.title("Customer Personality Analysis")
    st.header('Introduction')
    # Detailed summary about the backgroud of Customer Personality Analysis
    st.write("Customer personality analysis is the process of identifying and understanding the unique characteristics and traits that make up an individual customer's personality. This information can be used by companies to tailor their marketing and sales efforts to better target and serve each customer's specific needs and preferences.")
    st.write("Earlier Customer personality analysis was done manually by specific teams by identifying common patterns and trends among customers. However, Machine learning techniques has now made it possible to automate this process using algorithms that can analyze large amounts of data and identify common patterns and traits among customers.")
    st.write("One type of machine learning algorithm that can be used for customer personality analysis is unsupervised learning. The Clustering Unsupervised learning algorithms are trained on a large amount of data and can automatically detect patterns and similarities among customers without being explicitly told what to look for. This makes them particularly well suited for customer personality analysis, as they can uncover subtle differences and trends that might not be immediately apparent to humans.")
    st.write("Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. Finding the potential customers by analysing the behaviour of them is useful to understand the targeted customers. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyse which customer segment is most likely to buy the product and then market the product only on that particular segment.")
    st.header('About the data')
    st.write("The data consists of 2240 records with 29 columns or dimentions.")
    st.write("Following are the details pertaining to the 29 columns or dimentions.")
    st.markdown(
        """
        People
        - ID: Customer's unique identifier
        - Year_Birth: Customer's birth year
        - Education: Customer's education level
        - Marital_Status: Customer's marital status
        - Income: Customer's yearly household income
        - Kidhome: Number of children in customer's household
        - Teenhome: Number of teenagers in customer's household
        - Dt_Customer: Date of customer's enrollment with the company
        - Recency: Number of days since customer's last purchase
        - Complain: 1 if the customer complained in the last 2 years, 0 otherwise

        Products
        - MntWines: Amount spent on wine in last 2 years
        - MntFruits: Amount spent on fruits in last 2 years
        - MntMeatProducts: Amount spent on meat in last 2 years
        - MntFishProducts: Amount spent on fish in last 2 years
        - MntSweetProducts: Amount spent on sweets in last 2 years
        - MntGoldProds: Amount spent on gold in last 2 years

        Promotion
        - NumDealsPurchases: Number of purchases made with a discount
        - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
        - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
        - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
        - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
        - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
        - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise

        Place
        - NumWebPurchases: Number of purchases made through the company’s website
        - NumCatalogPurchases: Number of purchases made using a catalogue
        - NumStorePurchases: Number of purchases made directly in stores
        - NumWebVisitsMonth: Number of visits to company’s website in the last month

        Target
        - Need to perform clustering to summarize customer segments.
        """
)
    

    # ### **Importing data from file project-data.csv** ###
    # # Reading the data from the file 'project-data.csv'
    # ldc_orig_data = pd.read_csv("project-data.csv",delimiter=";")

    # Reading the data from the file 'marketing_campaign.xlsx'
    ccspa_orig_data = pd.read_excel("marketing_campaign.xlsx", parse_dates=['Dt_Customer'])

    # Reading the data from the file 'prediction.csv'
    prediction = pd.read_csv("prediction.csv")


    ''' The column name in the example case is "Unnamed: 0" but it works with any other name ("Unnamed: 0" for example). '''

    prediction.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True) 
    # Then, drop the column as usual.
    prediction.drop(["a"], axis=1, inplace=True)
    st.session_state['prediction'] = prediction
    
    # Creating a replicate 'ldc_data' of the original DataFrame 'ldc_orig_data'
    ccspa_data = ccspa_orig_data.copy()

    # Dropping the missing values
    ccspa_data['Income'] = ccspa_data['Income'].fillna(0)

    # Renaming column names
    ccspa_data.rename(columns = {'Year_Birth':'YOB','Marital_Status': 'MS'}, inplace = True)

    # Changing the 'Marital_Status' and 'Education' columns to categorical columns
    for col in ['MS', 'Education']:
        ccspa_data[col] = ccspa_data[col].astype('category')



    categorical_cols = []
    numeric_cols = []
    for columns in ccspa_data.columns:
        if ((ccspa_data[columns].dtypes == 'object') or (ccspa_data[columns].dtypes == 'category')):
                categorical_cols.append(columns)
        elif ((ccspa_data[columns].dtypes != 'object') and (ccspa_data[columns].dtypes != 'category')):
                numeric_cols.append(columns)
    
    # Creating dataframes for both categorical data as well as numeric data
    ccspa_data_cat = pd.DataFrame()
    ccspa_data_num = pd.DataFrame()

    ccspa_data_cat = ccspa_data[categorical_cols]
    ccspa_data_num = ccspa_data[numeric_cols]

    
    # st.session_state['categorical_cols'] = categorical_cols
    # st.session_state['numeric_cols'] = numeric_cols
    # st.session_state['ccspa_data_cat'] = ccsdata
    # st.session_state['ccspa_data_num'] = ccsdata

    ccs_data = ccspa_data.copy()
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()

    for col in ccspa_data_cat:
        ccs_data[str(col) + "_N"]= label_encoder.fit_transform(ccs_data[col])
    
    
    
    # Removing the Categorical Columns once we have desciphered the encoding done by LabelEncoder
    ccsdata = ccs_data.drop(['Education','MS', 'Z_Revenue', 'Z_CostContact', 'Dt_Customer'], axis = 'columns')
    st.session_state['ccsdata'] = ccsdata
    st.session_state['ccspa_data'] = ccspa_data
    # st.table(ccsdata)
    
    # Normalization of the data using MinMaxScaler Function
    scaler = MinMaxScaler()
    ccs_norm = pd.DataFrame()
    ccs_norm[['ID','Income','Kidhome','Teenhome','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Response','Education_N','MS_N']] = scaler.fit_transform(ccsdata[['ID','Income','Kidhome','Teenhome','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Response','Education_N','MS_N']])

    pca=PCA(n_components=10)
    pca.fit(ccs_norm)
    scores_pca=pca.transform(ccs_norm)
    st.session_state['scores_pca'] = scores_pca
    
    ccs_corr_data = pd.DataFrame()
    ccs_corr_data[['ID','Income','Kidhome','Teenhome','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Response']] = ccsdata[['ID','Income','Kidhome','Teenhome','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Response']]
    st.session_state['ccs_corr_data'] = ccs_corr_data
    

    


if __name__ == '__main__':
    main()
        

        
        
        
        
        
        
        