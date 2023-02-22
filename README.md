## Introduction

This repository contains a script and pipeline for stroke prediction model deployment. Model building and optimization process can be found in [
stroke_and_health_markers_prediction](https://github.com/LiucijaSvink/stroke_and_health_markers_prediction) repository. The app was designed to be used by healthcare providers or anybody else curious to find out whether they are at risk of stroke based on other basic information.

## Data

Data to build the model was taken from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

## Content

The repository contains 3 files:
- **requirements.txt** contains a list of necessary dependencies for model loading and deployment
- **stroke_model.pkl** is a file where the created and optimized stroke model is stored.
- **streamlit_app_script.py** contains a script to build the streamlit app, load the model and make predictions on new input data.

## App

To access the deployed model app in the streamlit platform, use the following [link](https://liucijasvink-stroke-prediction-app-streamlit-app-script-y5i2h6.streamlit.app/).

## Usage

Enter the information in respective app fields and click the button **Predict stroke**. The app will perform a prediction using the provided data and inform you whether the person is at risk of stroke or not.
