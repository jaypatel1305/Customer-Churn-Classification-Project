# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:47:19 2022

@author: Admin
"""

#pip install sreamlit

import streamlit as st
import pandas as pd
import pickle


st.title('Model Deployment on Telecom Churn dataset')



st.sidebar.header('User Input Parameters')




def user_input_features():
    sc = st.sidebar.selectbox('SeniorCitizen',(0,1))
    pr = st.sidebar.selectbox('Partner',('Yes','No'))
    dep = st.sidebar.selectbox('Dependents',('Yes','No'))
    
    ten = st.slider("Tenure",min_value=0,max_value=75,step=1)
    ml = st.sidebar.selectbox('MultipleLines',('No phone service', 'No', 'Yes'))
    isr = st.sidebar.selectbox('InternetService',('DSL', 'Fiber optic' ,'No'))
    osr = st.sidebar.selectbox('OnlineSecurity',('No', 'Yes' ,'No internet service'))
    ob = st.sidebar.selectbox('OnlineBackup',('No', 'Yes' ,'No internet service'))
    
    dp = st.sidebar.selectbox('DeviceProtection',('No', 'Yes' ,'No internet service'))
    stv = st.sidebar.selectbox('StreamingTV',('No', 'Yes' ,'No internet service'))
    sms = st.sidebar.selectbox('StreamingMovies',('No', 'Yes' ,'No internet service'))
    
    ts = st.sidebar.selectbox('TechSupport',('No', 'Yes' ,'No internet service'))
    
    cr = st.sidebar.selectbox('Contract',('Month-to-month', 'One year' ,'Two year'))
    
    
    pb = st.sidebar.selectbox('PaperlessBilling',('Yes','No'))
    pm = st.sidebar.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)',
       'Credit card (automatic)'))
    
    
    mc = st.sidebar.number_input("Insert the MonthlyCharges",min_value=10,max_value=1000,step=1)
    tc = st.sidebar.number_input("Insert TotalCharges",min_value=10,max_value=1000,step=1)
    
   
    
    data= {'SeniorCitizen': sc,
             'Partner': pr,
             'Dependents': dep,
             'tenure': ten,
             
             'OnlineSecurity': osr,
             'OnlineBackup': ob,
             'DeviceProtection':dp,
             'StreamingTV': stv,
             'StreamingMovies': sms,
             'TechSupport':ts,
             'Contract':cr,
             'PaperlessBilling': pb,
             'PaymentMethod': pm,
             'MonthlyCharges': mc,
             'TotalCharges': tc}
    
    

    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)



with open(file="final_model.sav",mode="rb") as f1:
    model = pickle.load(f1)
    
    
prediction = model.predict(df)
#prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')
#st.write('Yes' if prediction_proba[0][1] > 0.5 else 'No')

#st.subheader('Prediction Probability')
st.write(prediction[0])



#live surver: streamlit.io, heroku, AWS, Azure













