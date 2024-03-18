"""
Created on Sat Jan 20 10:42:05 2024

@author: Clive Dominic Andrews
"""

# import the libraries
import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

def main():
    
    st.title("Customer Personality Analysis")
    st.header('Clustering Project')

    # Importing the model
    import pickle
    pickle_in = open('KMEANS.pkl','rb')
    pickled_kmeans = pickle.load(pickle_in)

    # Code to extract data from another page based on data saved in a session.
    # Check if you've already initialized the data
    if 'ccsdata' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
    else:
        # Getting the latest Liver Disease Dataframe extracted in home page
        ccsdata = st.session_state['ccsdata']
    

    results = []

    # ID = st.number_input("**What's the ID?** :id:", min_value=0, max_value=15000, step= 1) # 2
    # YOB = st.date_input("**What is the YOB?** :birthday:", value=None , min_value=1700 , max_value=2024 , key=None)
    YOB = st.number_input("**What is the YOB?** :birthday:", min_value=1700, max_value=2024, step= 1)
    Income = st.number_input("**What is the INCOME?** :money_with_wings:", min_value=0, max_value=10000000, step= 1)
    Kidhome = st.number_input("**What is the count of Kids at home?** :baby:", min_value=0, max_value=20, step= 1)
    Teenhome = st.number_input("**What is the count of Teenagers at home?** :child:", min_value=0, max_value=20, step= 1)
    Recency = st.number_input("**What is the number of days since customer's last purchase?** :calendar:", min_value=0, max_value=100, step= 1)
    MntWines = st.number_input("**Amount spent on wine in last :two: years?** :wine_glass:", min_value=0, max_value=2000, step= 1)
    MntFruits = st.number_input("**Amount spent on fruits in last :two: years?** :green_salad:", min_value=0, max_value=200, step= 1)
    MntMeatProducts = st.number_input("**Amount spent on meat in last :two: years?** :poultry_leg: :cut_of_meat:", min_value=0, max_value=1800, step= 1)
    MntFishProducts = st.number_input("**Amount spent on fish in last :two: years?** :fish:", min_value=0, max_value=300, step= 1)
    MntSweetProducts = st.number_input("**Amount spent on sweets in last :two: years?** :candy:", min_value=0, max_value=300, step= 1)
    MntGoldProds = st.number_input("**Amount spent on gold in last :two: years?** :moneybag:" , min_value=0, max_value=380, step= 1)
    NumDealsPurchases = st.number_input("**Number of purchases made with a discount?** :money_mouth_face:", min_value=0, max_value=50, step= 1)
    NumWebPurchases = st.number_input("**Number of purchases made through the company’s website?** :computer:", min_value=0, max_value=50, step= 1)
    NumCatalogPurchases = st.number_input("**Number of purchases made using a catalogue?** :bookmark_tabs:", min_value=0, max_value=50, step= 1)
    NumStorePurchases = st.number_input("**Number of purchases made directly in stores** :shopping_trolley:", min_value=0, max_value=50, step= 1)
    NumWebVisitsMonth = st.number_input("**Number of visits to company’s website in the last month?** :computer: :calendar:", min_value=0, max_value=50, step= 1)
    Complain_f = st.radio("**If the customer complained in the last :two: years, 0 otherwise?** :person_with_pouting_face: :man-pouting:", ('Yes','No'),horizontal=True, index=1)

    # Checking conditions Accepted offers for Campaigns i.e., 1, 2, 3, 4 or 5
    if Complain_f == 'Yes':
        Complain = 1
    else:
        Complain = 0

    Response_f = st.radio("**If customer accepted the offer in the last campaign, 0 otherwise?** :raised_hands:", ('Yes','No'),horizontal=True, index=1)

    # Checking conditions Accepted offers for Campaigns i.e., 1, 2, 3, 4 or 5
    if Response_f == 'Yes':
        Response = 1
    else:
        Response = 0

    Education_f = st.selectbox("**What is the Customer's education level?** :books: :male-student:",
    ('2n Cycle','Basic','Graduation','Master','PhD'))

    # Checking conditions for Education
    if Education_f == '2n Cycle':
        Education_N = 0
    elif Education_f == 'Basic':
        Education_N = 1
    elif Education_f == 'Graduation':
        Education_N = 2
    elif Education_f == 'Master':
        Education_N = 3
    elif Education_f == 'PhD':
        Education_N = 4
    
    MS_f = st.selectbox("**What is the Customer's marital status?**:woman-raising-hand::man-raising-hand::man_and_woman_holding_hands::women_holding_hands::two_men_holding_hands::family:",
    ('Absurd','Alone','Divorced','Married','Single','Together','Widow','YOLO'))

    # Checking conditions for Marital Status
    if MS_f == 'Absurd':
        MS_N = 0
    elif MS_f == 'Alone':
        MS_N = 1
    elif MS_f == 'Divorced':
        MS_N = 2
    elif MS_f == 'Married':
        MS_N = 3
    elif MS_f == 'Single':
        MS_N = 4
    elif MS_f == 'Together':
        MS_N = 5
    elif MS_f == 'Widow':
        MS_N = 6
    elif MS_f == 'YOLO':
        MS_N = 7

    AcceptedCmp_1 = st.radio("**Did the customer accepted the offer in the :one: st campaign?**", ('Yes','No'),horizontal=True, index=1)
    AcceptedCmp_2 = st.radio("**Did the customer accepted the offer in the :two: nd campaign?**", ('Yes','No'),horizontal=True, index=1)
    AcceptedCmp_3 = st.radio("**Did the customer accepted the offer in the :three: rd campaign?**", ('Yes','No'),horizontal=True, index=1)
    AcceptedCmp_4 = st.radio("**Did the customer accepted the offer in the :four: th campaign?**", ('Yes','No'),horizontal=True, index=1)
    AcceptedCmp_5 = st.radio("**Did the customer accepted the offer in the :five: th campaign?**", ('Yes','No'),horizontal=True, index=1)

    # Checking conditions Accepted offers for Campaigns i.e., 1, 2, 3, 4 or 5
    if AcceptedCmp_1 == 'Yes':
        AcceptedCmp1 = 1
    else:
        AcceptedCmp1 = 0

    if AcceptedCmp_2 == 'Yes':
        AcceptedCmp2 = 1
    else:
        AcceptedCmp2 = 0

    if AcceptedCmp_3 == 'Yes':
        AcceptedCmp3 = 1
    else:
        AcceptedCmp3 = 0

    if AcceptedCmp_4 == 'Yes':
        AcceptedCmp4 = 1
    else:
        AcceptedCmp4 = 0

    if AcceptedCmp_5 == 'Yes':
        AcceptedCmp5 = 1
    else:
        AcceptedCmp5 = 0


    results = [[YOB, Income, Kidhome, Teenhome, Recency, MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds, NumDealsPurchases, NumWebPurchases, NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response, Education_N, MS_N]]


    if st.button('Submit'):
        prediction = pickled_kmeans.predict(results)
        if (prediction == 0):
            st.write("The Customer's personality falls in **'Cluster 0'**.")
        elif (prediction == 1):
            st.write("The Customer's personality falls in **'Cluster 1'**.")
        elif (prediction == 2):
            st.write("The Customer's personality falls in **'Cluster 2'**.")
        elif (prediction == 3):
            st.write("The Customer's personality falls in **'Cluster 3'**.")   



if __name__ == '__main__':
    main()

        
        

        
        
        
        
        
        
        