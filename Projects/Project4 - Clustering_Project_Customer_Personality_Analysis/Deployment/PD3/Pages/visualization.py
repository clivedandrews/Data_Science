"""
Created on Sat Mar 09 15:16:05 2024

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
    
    st.title("CUSTOMER PERSONALITY ANALYSIS")
    st.header('Clustering Project Visualization')

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

    
    if 'scores_pca' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
    else:
        # Getting the latest Liver Disease Normalized Dataframe from in home page
        scores_pca = st.session_state['scores_pca']

    if 'prediction' not in st.session_state:
        # Get the data if you haven't
        st.error('Please go back to home page and select the data.')
    else:
        # Getting the latest Liver Disease Normalized Dataframe from in home page
        prediction = st.session_state['prediction']
    


    # km_p_predict=pickled_kmeans.fit_predict(ccsdata)

    X = scores_pca
    kmeans_plot = plt.figure(figsize=(10,4))
    plt.scatter(X[prediction['KM_P']==0, 0], X[prediction['KM_P']==0, 1], s=100, c='orange', label ='Cluster 1')
    plt.scatter(X[prediction['KM_P']==1, 0], X[prediction['KM_P']==1, 1], s=100, c='blue', label ='Cluster 2')
    plt.scatter(X[prediction['KM_P']==2, 0], X[prediction['KM_P']==2, 1], s=100, c='green', label ='Cluster 3')
    plt.scatter(X[prediction['KM_P']==3, 0], X[prediction['KM_P']==3, 1], s=100, c='red', label ='Cluster 4')
    plt.title("K-MEANS CLUSTERING USING PIPELINE\n", fontweight="bold", horizontalalignment="center", fontstyle = "normal", fontsize = "18")
    # Function add a legend
    plt.legend(loc="lower right")
    plt.show()
    plt.show(block=False)
    st.pyplot(kmeans_plot)

    X = scores_pca
    kmeans_plot = plt.figure(figsize=(10,4))
    plt.scatter(X[prediction['HC_P']==0, 0], X[prediction['HC_P']==0, 1], s=100, c='orange', label ='Cluster 1')
    plt.scatter(X[prediction['HC_P']==1, 0], X[prediction['HC_P']==1, 1], s=100, c='blue', label ='Cluster 2')
    plt.scatter(X[prediction['HC_P']==2, 0], X[prediction['HC_P']==2, 1], s=100, c='green', label ='Cluster 3')
    plt.scatter(X[prediction['HC_P']==3, 0], X[prediction['HC_P']==3, 1], s=100, c='red', label ='Cluster 4')
    plt.title("HIERARCHIAL CLUSTERING USING PIPELINE\n", fontweight="bold", horizontalalignment="center", fontstyle = "normal", fontsize = "18")
    # Function add a legend
    plt.legend(loc="lower right")
    plt.show()
    plt.show(block=False)
    st.pyplot(kmeans_plot)

    X = scores_pca
    kmeans_plot = plt.figure(figsize=(10,4))
    plt.scatter(X[prediction['DB_P']==0, 0], X[prediction['DB_P']==0, 1], s=100, c='orange', label ='Cluster 1')
    plt.scatter(X[prediction['DB_P']==1, 0], X[prediction['DB_P']==1, 1], s=100, c='blue', label ='Cluster 2')
    plt.scatter(X[prediction['DB_P']==2, 0], X[prediction['DB_P']==2, 1], s=100, c='green', label ='Cluster 3')
    plt.scatter(X[prediction['DB_P']==3, 0], X[prediction['DB_P']==3, 1], s=100, c='red', label ='Cluster 4')
    plt.title("DBSCAN CLUSTERING USING PIPELINE\n", fontweight="bold", horizontalalignment="center", fontstyle = "normal", fontsize = "18")
    # Function add a legend
    plt.legend(loc="lower right")
    plt.show()
    plt.show(block=False)
    st.pyplot(kmeans_plot)



if __name__ == '__main__':
    main()
