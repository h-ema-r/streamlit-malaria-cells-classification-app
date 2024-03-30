## Malaria Detection Streamlit App

## Problem Statement:
The detection of malaria-infected cells in microscopic images is a critical task in healthcare, as it aids in the timely diagnosis and treatment of malaria. However, manual examination of these images is labor-intensive and prone to errors. Therefore, there is a need for automated systems that can accurately classify malaria-infected cells from uninfected ones.

## Goal
Our goal is to develop a robust deep learning model capable of accurately detecting malaria-infected cells in microscopic images. Additionally, we aim to create a user-friendly application using Streamlit that allows users to upload images for classification. The application will not only provide predictions but also visualize the areas within the images that influence the model's decision-making process, using Explainable AI techniques such as heatmap visualization. By achieving this goal, we aim to improve the efficiency and accuracy of malaria diagnosis, ultimately contributing to better healthcare outcome.

## Data source 
The dataset used for training the model is available at [Data source](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)

## Files:
**requirements.txt:**
This file contains all the dependencies required to run the application.

**models1/malaria-detection.ipynb**
- This notebook contains the code for training a malaria detection model using Convolutional Neural Network (CNN).<br>
- Once training is completed, the trained model is saved to a file using following command for future use.<br>
```bash
model.save('malaria-detection-model.h5')
```

**app.py:**
- This file contains code for a Streamlit app.<br>
- The app utilizes the heatmap visualization technique to provide insights into how the model analyzes input images, aiding in the interpretation of its predictions. <br>
- By overlaying the heatmap on the original image, users can visually identify the regions that the model considered most important for its classification decision, which will be colored red in our output.



# Usage

## Creating a virtual Environment
```bash
python -m virtualenv malaria-env
```

## Activating the Virtual Environment 
```bash
malaria-env/Scripts/Activate.ps1
```
## Installing Necessary Requirements
```bash
python -r requirements.txt
```
## Running the Streamlit App
```bash 
streamlit run app.py
```

## App working Video

Below are the videos demonstrating the Malaria Detection Streamlit App:

![Video 1:](malaria-detection-streamlit-app1.mp4)

<br>

![Video 2:](malaria-detection-streamlit-app2.mp4)
