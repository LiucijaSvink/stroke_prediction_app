# -*- coding: utf-8 -*-
"""
@author: Liucija Svinkunaite
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Loading up the classification pipeline created
stroke_model = joblib.load('stroke_model.pkl')
threshold = 0.2967177

# Caching the model for faster loading
@st.cache

# Define prediction function
def prepare_data(gender: str, age: float, heart_disease: str, ever_married:str,
                 work_type: str, avg_glucose_level: float, bmi: float):
    '''Convert streamlit app data to a suitable input format for the stroke 
    model pipeline to derive a prediction'''
    
    if gender == 'other':
        gender = np.nan
    
    if work_type == 'government job':
        work_type = 'govt_job'
    elif work_type == 'job in private sector':
        work_type = 'private'
    elif work_type == 'self-employed':
        work_type = 'private'
    elif work_type == 'taking care of children':
        work_type = 'children'
    elif work_type == 'never worked':
        work_type = 'never_worked'
    elif work_type == 'never worked':
        work_type = 'never_worked'
    elif work_type == 'self_employed':
        work_type = work_type
    
    # Derive stroke probability
    if age < 45: 
        stroke_prob = 0.6
    else:
        stroke_prob =  0.6 * 2 ** ((age-45)/10)
    
    # Assign bmi category
    if bmi <= 18:
        bmi_category = '1'
    elif bmi > 18 and bmi <= 25:
        bmi_category = '2'
    elif bmi > 25 and bmi <= 30:
        bmi_category = '3'
    elif bmi > 30 and bmi <= 35:
        bmi_category = '4'
    elif bmi > 35 and bmi <= 120:
        bmi_category = '5'
    
    if avg_glucose_level >= 126:
        diabetes = 1
    else:
        diabetes = 0
    
    column_names = ['gender', 'age', 'heart_disease', 'ever_married', 
                    'work_type', 'avg_glucose_level', 'bmi', 'stroke_prob', 
                    'bmi_category', 'diabetes']
    
    data = pd.DataFrame([[gender, age, heart_disease, ever_married, work_type,
                          avg_glucose_level, bmi, stroke_prob, bmi_category, 
                          diabetes]], columns=column_names)
    
    # Assign respective data types 
    categorical_cols = ['gender', 'heart_disease', 'ever_married', 
                       'work_type', 'bmi_category', 'diabetes']

    data[categorical_cols] = data[categorical_cols].astype('category')
    
    return data
        
def make_prediction(pipeline: Pipeline, data: pd.DataFrame, threshold: float):
    '''Using the input data, derive a prediction whether the patient is likely
    to get a stroke or not'''
    
    predicted_prob = pipeline.predict_proba(data)[:, 1]
    prediction = (predicted_prob >= threshold).astype(int)
    if prediction == 1:
        prediction = 'Patient is likely to experience stroke.'
    elif prediction == 0:
        prediction = 'Patient is not likely to experience stroke'
            
    return prediction


st.title('Stroke prediction')
st.header('Please enter the data of the patient:')
gender = st.selectbox('Gender:', ['female', 'male'])
age = st.number_input('Age:', min_value=0.1, max_value=110.0, value=30.0)
heart_disease = st.selectbox('Does the patient have heart disease?', ['yes', 'no'])
ever_married = st.selectbox('Was the patient ever married?', ['yes', 'no'])
work_type = st.selectbox('What type of job does the patient have?', 
                         ['government job', 'job in private sector', 
                          'self-employed', 'taking care of children',
                          'never worked'])

avg_glucose_level = st.number_input('Enter the average glucose level of the \
                                    patient:', min_value=40.0, max_value=300.0, 
                                    value=100.0)

bmi = st.number_input('Enter BMI of the patient:', min_value=14.0,
                      max_value=300.0, value=100.0)

if st.button('Predict stroke'):
    stroke_data = prepare_data(gender, age, heart_disease, ever_married, 
                               work_type, avg_glucose_level, bmi)
    
    stroke_prediction = make_prediction(stroke_model, stroke_data, threshold)
    st.success(f'{stroke_prediction}')