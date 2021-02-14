# Project: Backorder Prediction using Machine Learning Classifiers
This project concerns a binary classification problem. Particularly, a backorder is the order which could not be fulfilled by the company. Due to high demand of a product, the company was not able to keep up with the delivery of the order. The backordering can lead to upsetting customer as they couldn't get what they ordered and the loyalty will decrease.
Also, company cannot overstock every product in their inventory to avoid such situation.
There has to be a way for the company to know for which products they can face this problem.
So, the company has shared a data file with different input features for each product and it hopes to find a pattern inside this data which can give them some insight.
The data file contains the historical data for some weeks prior to the week we are trying to predict.
The data has 23 columns including 22 features and one target column.
The dataset used can be found here: [Dataset Link](https://www.dropbox.com/s/mh554ii745vmu8y/backorder%20prediction.csv?dl=0).
The code implemented in Python 3.6.9 and the following libraries are used: sklearn,numpy,pandas,imblearn,matplotlib
This repo contains a single Google Colab (Python) 
# Google Colab Notebook Sections
## Imports
* sklearn for Machine Learning
* imblearn
* numpy for Matrices
* scipy for Stats
* pandas for Dataframe manipulation
* matplotlib, plotly and seaborn for Visualization
## Data Preparation/exploration
* Check dataset for nullity or negative values
* Plot correlation matrix for correlation coefficients between variables
## Classification Models
* Support Vector Machine
* Random Forest
* KNearest Neighbors
## Metrics 
In this project the following metrics are used for evaluation
* Precision
* Recall
* F1 score
## Removing Ouliers and Tuning Hyparameters


# Proccess Description
The experiment can be divide into the following steps:
1. Split the initial dataset in train and test datasets
2. Impute and scale the training dataset(fit_transform train dataset and transform the test dataset)
3. Apply SMOTE only in training dataset
4. Fit the estimators and predict the classes
5. Plot the confusion matrix for each classifier and precision-recall function
6. Detect ouliers and remove them
7. Tune the hyparameters of each classifiers
8. Fit the estimators and predict the classes after tuning
9. Plot the confusion matrix for each classifier and precision-recall function after tuning

# Results
4. 
