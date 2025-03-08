# Project-4
Deep CNN image classifier
-->Project Overview
We are building here a deep convolutional neural network Image classifier. What it does is basically we develop a model that takes in images of happy and sad people and  trained on a given dataset, It predicts whether the photo given by the user is a happy photo or a sad photo. We can also expand it's use case to express emotions of anger, fear, surprise and disgust. It analyzes the image given by user and generates a value on a certain index and compares the value if it falls below a threhold it tells it sad/happy and vice versa.
-->Installation Instructions
Must have a programming application. Ex-Python.
import libraries like tensorflow,numpy, matplotlib, OpenCv,cv2 and keras
Should have a GPU
-->Model Training Steps
First of all build a deep learning model
Train it using keras callbacks functions.
Do fitting for the model and providing epochs.
Plot the performance and check whether model working accurate or not.
-->Dataset Information
Dwownload tonnes of images from Web and remove the dodgy ones so that model works accurately.  Then load it into the program using a pipeline.
-->Example Model Predictions
Test the model's accuracy by providing it with a happy image. If it gives output as happy image means the model is working correctly.
