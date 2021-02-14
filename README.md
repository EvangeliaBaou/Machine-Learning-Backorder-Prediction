# Project: Backorder Prediction using Machine Learning Classifiers
This project concerns a binary classification problem. Particularly, a backorder is the order which could not be fulfilled by the company. Due to high demand of a product, the company was not able to keep up with the delivery of the order. The backordering can lead to upsetting customer as they couldn't get what they ordered and the loyalty will decrease.
Also, company cannot overstock every product in their inventory to avoid such situation.
There has to be a way for the company to know for which products they can face this problem.
So, the company has shared a data file with different input features for each product and it hopes to find a pattern inside this data which can give them some insight.
The data file contains the historical data for some weeks prior to the week we are trying to predict.
The data has 23 columns including 22 features and one target column.
The dataset used can be found here: [Dataset Link](https://www.dropbox.com/s/mh554ii745vmu8y/backorder%20prediction.csv?dl=0).
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
## Removing Outliers and Tuning Hyparameters


# Proccess Description
The experiment can be divide into the following steps:
1. Split the initial dataset in train and test 
2. Impute and scale the training dataset(fit_transform train dataset and transform the test dataset)
3. Apply SMOTE only in training dataset
4. Fit the estimators and predict the classes
5. Plot the confusion matrix for each classifier and precision-recall function
6. Detect ouliers and remove them
7. Tune the hyparameters of each classifiers
8. Fit the estimators and predict the classes after tuning
9. Plot the confusion matrix for each classifier and precision-recall function after tuning

# Results
## Before Tuning and Outlier removal
### SVM

              precision    recall  f1-score   support

           0       0.78      0.98      0.87      4756
           1       0.71      0.14      0.23      1553

    accuracy                            0.77     6309
    macro avg       0.75      0.56      0.55     6309
    weighted avg    0.76      0.77      0.71     6309

### Random Forest
              precision    recall  f1-score   support

           0       0.94      0.94      0.94      4730
           1       0.83      0.81      0.82      1579

    accuracy                           0.91      6309
    macro avg       0.88      0.88     0.88     6309
    weighted avg    0.91      0.91     0.91     6309

### KNearest Neighbors
              precision    recall  f1-score   support

           0       0.87      0.88      0.87      4730
           1       0.63      0.62      0.62      1579

    accuracy                           0.81      6309
    macro avg       0.75      0.75      0.75     6309
    weighted avg    0.81      0.81      0.81     6309


## After Tuning and Outlier removal
### SVM
              precision    recall  f1-score   support

           0       0.88      0.94      0.91      4424
           1       0.76      0.61      0.68      1421

    accuracy                           0.86      5845
    macro avg       0.82      0.78     0.79      5845
    weighted avg    0.85      0.86     0.85      5845
    
 ## Random Forest
               precision    recall  f1-score   support

           0       0.95      0.95      0.95      4797
           1       0.83      0.84      0.83      1512

    accuracy                           0.92     6309
    macro avg       0.89      0.89     0.89     6309
    weighted avg    0.92      0.92     0.92     6309
    
   ## KNearest Neighbors
              precision    recall  f1-score   support

           0       0.91      0.88      0.89      4424
           1       0.65      0.72      0.69      1421

    accuracy                            0.84     5845
    macro avg       0.78      0.80      0.79     5845
    weighted avg    0.85      0.84      0.84     5845


## Conclusion
The first experiments without outliers' removal suggest a significant difference between KNN’s, RF’s and SVM’s performance, with the latter indicating the worst results.
After outlier removal and parameter tuning, SVM indicates a remarkable improvement while Random forest and KNN have slightly better performance. However, Random Forest remains the best classifier in terms of f1-score.

## Future Work
* Evaluation of SVM and KNN with Feature selection 
* Evaluation of the models for various ratios of SMOTE
