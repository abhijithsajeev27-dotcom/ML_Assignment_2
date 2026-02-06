ML Assignment 2 – Breast Cancer Diagnosis using Machine Learning
a. Problem Statement

The objective of this assignment is to build and evaluate multiple machine learning classification models to predict whether a breast tumor is benign or malignant based on diagnostic features extracted from digitized images of breast mass biopsies. The task involves training several classifiers, comparing their performance using standard evaluation metrics, and deploying the trained models in a Streamlit web application for interactive prediction and evaluation.

b. Dataset Description

The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset, sourced from the UCI Machine Learning Repository.

Number of instances: 569

Number of features: 30 numerical features

Target variable:

0 → Benign

1 → Malignant

Feature types: Continuous diagnostic measurements such as radius, texture, perimeter, area, smoothness, concavity, symmetry, and fractal dimension.

This dataset is widely used for benchmarking binary classification algorithms in medical diagnosis tasks.

c. Models Used and Performance Comparison 

Six different machine learning models were trained and evaluated using an 80/20 stratified train-test split.
All metrics were calculated on the test dataset.

Comparison Table

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression      | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861   | 0.9623 |
| Decision Tree            | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286   | 0.8174 |
| kNN                      | 0.9737   | 0.9884 | 0.9600    | 1.0000 | 0.9796   | 0.9442 |
| Naive Bayes              | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517   | 0.8676 |
| Random Forest (Ensemble) | 0.9474   | 0.9937 | 0.9583    | 0.9583 | 0.9583   | 0.8869 |
| XGBoost (Ensemble)       | 0.9561   | 0.9950 | 0.9467    | 0.9861 | 0.9660   | 0.9058 |


d. Model Performance Observations
ML Model Name	Observation about Model Performance
Logistic Regression	Achieved the highest overall performance with excellent accuracy, AUC, and MCC, indicating that the dataset is largely linearly separable and well-suited for this model.
Decision Tree	Showed reasonable accuracy but lower AUC and MCC compared to other models, suggesting sensitivity to overfitting and limited generalization.
kNN	Delivered very high recall and strong overall performance, but its effectiveness depends on proper feature scaling and choice of neighbors.
Naive Bayes	Performed well despite strong independence assumptions, though its MCC was lower than ensemble models, indicating some misclassification.
Random Forest (Ensemble)	Provided robust and stable performance across metrics by reducing variance through ensemble averaging.
XGBoost (Ensemble)	Achieved excellent AUC and high MCC, demonstrating strong discriminative power and effective handling of feature interactions.
Summary

The results show that Logistic Regression, XGBoost, and Random Forest are the top-performing models for this dataset. Ensemble methods generally outperform single classifiers, while simpler models like Logistic Regression remain highly competitive due to the structured nature of the dataset. Multiple evaluation metrics, including MCC and AUC, provide a more comprehensive understanding of model performance, especially for medical diagnosis applications.
