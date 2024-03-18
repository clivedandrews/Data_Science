"""
Created on Sat Jan 20 10:42:05 2024

@author: Clive Dominic Andrews
"""

# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, scale
import xgboost as xgb
# import tf

import streamlit as st

import warnings
warnings.filterwarnings('ignore')
palette_color = sns.color_palette('rainbow')
scaler = MinMaxScaler()


    

def main():

    st.title("Liver Disease Classification")
    st.header('Introduction')
    # Detailed summary about Infosys
    st.write("In today's world, more than a million people are diagnosed with liver disease each year. Liver cirrhosis, hepatitis (A, B, C), liver cancer are common liver diseases. Globally, 1.32 million more people died of liver cirrhosis in 2017 than in 1990, of which 66.7% were men and 33.3% were women. Although overall death is declining because of improvements in advanced treatment and maintaining a sound lifestyle. However, liver-related fatalities accounted for 3.5% of all deaths this century. Excessive uses of drugs, alcohol, obesity, and diabetes are the main causes of liver disease. These life-threatening diseases are manageable if they are diagnosed in their early stages.")
    st.write("Machine learning techniques are widely used in the healthcare sector, in particular for the diagnosis and classification of certain diseases based on characteristic information. These systems will help clinicians make accurate decisions about patients. The input raw feature space is typically saturated with a significant amount of irrelevant feature information and frequently exhibits high dimensionality when the data is acquired using feature generation techniques in conventional ML systems. Projection-based statistical methods such as principal component analysis (PCA), factor analysis (FA), and linear discriminant analysis (LDA) work well to reduce dimensionality. PCA reduces the dimensionality of the dataset without losing significant feature information. Factor analysis is an extension of PCA that describes the covariance relationships between variables in terms of some underlying factors. LDA uses the class label to compute the matrix between and within the class and seeks the directions along which the classes are best separated. However, projection-based feature extraction, how many components should be retained is still an unsolved issue. ")
    st.write("To achieve better liver patient prediction using a computer-assisted diagnosis process we are accounting for all the stated data preprocessing techniques. The proposed integrated method enhances liver disease classification accuracy, prevents misdiagnosis of liver disease, and increases patient survival. Our project is to propose a statistical feature integration approach that aims to improve AUC and accuracy.")
    st.write("The healthy liver plays more than 500 organic roles in the human body, while a malfunction may be dangerous or even deadly. Early diagnosis and treatment of liver disease can improve the likelihood of survival. Machine learning (ML) is a powerful tool that can assist healthcare professionals during the diagnostic process for a hepatic patient. The standard ML system includes the methods of data pre-processing, feature extraction, and classification.")
    
    

    ### **Importing data from file project-data.csv** ###
    # Reading the data from the file 'project-data.csv'
    ldc_orig_data = pd.read_csv("project-data.csv",delimiter=";")
    
    # Creating a replicate 'ldc_data' of the original DataFrame 'ldc_orig_data'
    ldc_data = ldc_orig_data.copy()

    # Renaming column names
    ldc_data.rename(columns = {'category':'CATEGORY','age': 'AGE','sex':'SEX','albumin':'ALBUMIN','alkaline_phosphatase':'ALKAL_PHOS','alanine_aminotransferase':'ALAN_AMIN','aspartate_aminotransferase':'ASPA_AMINO','bilirubin':'BILIRUBIN','cholinesterase':'CHOLI','cholesterol':'CHOLESTEROL','creatinina':'CREATININA','gamma_glutamyl_transferase ':'GAMMA_GLU_TRAN','protein   ':'PROTEIN'}, inplace = True)

    # Dropping the missing values
    ldc_data = ldc_data.dropna()
    
    # Replacing space in the Category Column
    ldc_data['CATEGORY'] = ldc_data['CATEGORY'].str.replace(' ', '')

    # Changing the 'PROTEIN' column as numeric i.e., float64
    ldc_data['PROTEIN'] = pd.to_numeric(ldc_data['PROTEIN'])

    # Printing a liver disease categories
    st.header('LIVER DISEASE CATEGORIES:')
    category_list = []
    category_list = ldc_data.CATEGORY.unique()
    cat_list = pd.DataFrame()
    cat_list['LIVER DISEASE CATEGORIES'] = category_list
    st.table(cat_list)

    st.session_state['LDC_INIT'] = ldc_data
    # Segregating categorical Columns and numerical columns
    categorical_cols = []
    numeric_cols = []
    for columns in ldc_data.columns:
      if ((ldc_data[columns].dtypes == 'object') or (ldc_data[columns].dtypes == 'category')):
            categorical_cols.append(columns)
      elif ((ldc_data[columns].dtypes != 'object') and (ldc_data[columns].dtypes != 'category')):
            numeric_cols.append(columns)
    
    # Creating dataframes for both categorical data as well as numeric data
    ldc_cat_comp_data = pd.DataFrame()
    ldc_num_comp_data = pd.DataFrame()

    ldc_cat_comp_data = ldc_data[categorical_cols]
    ldc_num_comp_data = ldc_data[numeric_cols]

    # Label Encoding
    ldcdata = ldc_data.copy()
    from sklearn import preprocessing
    label_encoder = preprocessing.LabelEncoder()

    for col in ldc_cat_comp_data:
        ldcdata[str(col) + "_N"]= label_encoder.fit_transform(ldcdata[col])

   

    # Removing the Categorical Columns once we have desciphered the encoding done by LabelEncoder
    ldcdata = ldcdata.drop(['CATEGORY','SEX'], axis = 'columns')

    st.session_state['LDC_LED'] = ldcdata

    # Printing a Columns for Assessment of Disease
    st.header('Parameters reviewed for Liver Disease assessment')
    x_columns = ldcdata.drop(['CATEGORY_N'], axis=1).columns.tolist()
    ldf_col = pd.DataFrame()
    ldf_col['PARAMETERS'] = x_columns
    st.table(ldf_col)


    # Printing a sample table
    st.header('Sample Liver Disease Data')
    st.table(ldcdata.head())
    

    # After decoded the data we can now only consider the label encoded columns and not the categorical column 'SEX' and also exclude the target column 'CATEGORY_N'
    inputs_ldc= ldcdata.drop(['CATEGORY_N'], axis = 'columns')
    target_ldc = ldcdata['CATEGORY_N']

    # Normalization of Data
    norm_scaler = MinMaxScaler()
    ldc_norm = pd.DataFrame()
    
    ldc_norm[x_columns] = norm_scaler.fit_transform(ldcdata[x_columns])

    ldc_df = ldcdata.copy()

    st.session_state['LDC_NORM'] = ldc_norm

    # Scaling the data using StandardScaler
    from sklearn.preprocessing import StandardScaler
    std_scaler = StandardScaler()
    ldc_ss = pd.DataFrame()
    x_train_s = pd.DataFrame()
    x_test_s = pd.DataFrame()
    x_columns = ldcdata.drop(['CATEGORY_N'], axis=1).columns.tolist()
    ldc_ss[x_columns] = std_scaler.fit_transform(ldcdata[x_columns])
    st.session_state['LDC_STD'] = ldc_ss
    
    

    


if __name__ == '__main__':
    main()
        

        
        
        
        
        
        
        