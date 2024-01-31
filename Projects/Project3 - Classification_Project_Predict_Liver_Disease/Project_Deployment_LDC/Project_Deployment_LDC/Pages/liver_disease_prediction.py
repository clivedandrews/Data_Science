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
    
    st.title("Liver Disease Prediction")
    st.header('Just Some Random Header:')

    # Importing the model
    import pickle
    pickle_in = open('RF_MODEL.pkl','rb')
    pickled_rf_xgb_model = pickle.load(pickle_in)


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
    
    if 'LDC_STD' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
    else:
        # Getting the latest Liver Disease Standardized Dataframe from in home page
        ldc_ss = st.session_state['LDC_STD']

   

    # Importing the model
    import pickle
    pickled_rf_xgb_model = pickle.load(open('RF_MODEL.pkl','rb'))

    results = []

    Sex_Label = st.radio("What's the Sex?", ["**Male**:man:", "**Female**:woman:"], index=0,)
    if Sex_Label == "**Male**:man:":
        Sex = 1
    elif Sex_Label == '**Female**:woman:':
        Sex = 0
    Age = st.number_input("What's the Age?", min_value=1, max_value=130, step= 1) # 2
    Albumin = st.number_input("What is the value for Albumin?")
    Alkaline_Phosphatase = st.number_input("What is the value for Alkaline_Phosphatase?")
    Alanine_Aminotransferase = st.number_input("What is the value for Alanine_Aminotransferase?")
    Aspartate_Aminotransferase = st.number_input("What is the value for Aspartate_Aminotransferase?")
    Bilirubin = st.number_input("What is the value for Bilirubin?")
    Cholinesterase = st.number_input("What is the value for Cholinesterase?")
    Cholesterol = st.number_input("What is the value for Cholesterol?")
    Creatinina = st.number_input("What is the value for Creatinina?")
    Gamma_Glutamyl_Transferase = st.number_input("What is the value for Gamma_Glutamyl_Transferase?")
    Protein = st.number_input("What is the value for Protein?")

    results = [[Age, Albumin, Alkaline_Phosphatase, Alanine_Aminotransferase, Aspartate_Aminotransferase, Bilirubin, Cholinesterase, Cholesterol, Creatinina, Gamma_Glutamyl_Transferase, Protein, Sex]]


    if st.button('Submit'):
        xgb_prediction = pickled_rf_xgb_model.predict(results)
        if (xgb_prediction == 0):
            st.error("You have 'Cirrhosis'.")
        elif (xgb_prediction == 1):
            st.error("You have 'Fibrosis'.")
        elif (xgb_prediction == 2):
            st.error("You have 'Hepatitis'.")
        elif (xgb_prediction == 3):
            st.success("You do not have any Liver Disease.")
        elif (xgb_prediction == 4):
            st.error("You are suspected to have a Liver Disease.")    




if __name__ == '__main__':
    main()

        
        

        
        
        
        
        
        
        