import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler 

# 1. Set up the web page title
st.title("EV Charging Demand")
st.write("Enter the required inputs below to get a prediction from the trained model.")

# Load the model from file

with open("D:\Projects-Guvi\EV\lrg.pkl", "rb") as file:
        lrg = pickle.load(file)

with open(r"D:\Projects-Guvi\EV\rforest.pkl", "rb") as file:
        rforest = pickle.load(file)

st.sidebar.title("Regression and Classification")
selection = st.sidebar.radio('Go to', ['Home', 'Regression', 'classification'])
if selection == 'Home':
    st.title('Welcome')

elif selection == 'Regression':
    st.title('Regression')


#creating Columns:

    col = st.columns(1)
    UserID = st.number_input ('User ID')
    ChargerID = st.number_input ('Charger ID')
    ChargerCompany = st.number_input ('Charger Company')
    ChargerType = st.number_input ('Charger Type')
    Duration = st.number_input ('Duration')
    Log_Demand = st.number_input('Log_Demand')
    Duration_days = st.number_input('Duration_days')
    charging_time = st.number_input('charging_time')

    data = ( UserID,ChargerID,ChargerCompany,ChargerType,Duration,Log_Demand,Duration_days,charging_time)
         
    if data and st.button('PREDICT DEMAND'):
    
            
            input_data = np.array(data).reshape(1, -1)
           
            # Make predictions
            prediction = lrg.predict(input_data)
            st.write(prediction)

elif selection == 'classification':
    st.title('Classification')
            
#creating Columns:

    col = st.columns(1)
    UserID = st.number_input ('User ID')
    ChargerID = st.number_input ('Charger ID')
    ChargerCompany = st.number_input ('Charger Company')
    ChargerType = st.number_input ('Charger Type')
    Duration = st.number_input ('Duration')
    

    data = (UserID,ChargerID,ChargerCompany,ChargerType,Duration)
         
    if data and st.button('PREDICT DEMAND'):
    
            
            input_data = np.array(data).reshape(1, -1)
           
            # Make predictions
            prediction = rforest.predict(input_data)
            st.write(prediction)



