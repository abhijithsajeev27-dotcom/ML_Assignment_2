📄 Breast Cancer Classification using Machine Learning
1. Problem Statement

The objective of this project is to build and evaluate multiple machine learning models to classify breast tumors as Malignant (0) or Benign (1) using the Wisconsin Breast Cancer dataset.

Early detection of malignant tumors is critical in medical diagnostics, as delayed identification can lead to severe health consequences. Therefore, this project focuses on developing reliable classification models and comparing their performance using multiple evaluation metrics.

2. Dataset Description 

The dataset used in this project is the Wisconsin Breast Cancer Dataset, available via sklearn.datasets.

Dataset Characteristics:

Total Samples: 569

Features: 30 numerical features

Target Classes:

0 → Malignant

1 → Benign

The 30 features represent statistical properties (mean, standard error, worst case) of cell nuclei extracted from digitized breast mass images, including:

Radius

Texture

Perimeter

Area

Smoothness

Compactness

Concavity

Symmetry

Fractal dimension

The goal is to predict whether a tumor is malignant or benign based on these measurements.

3. Models Used 

The following six classification models were trained and evaluated:

Logistic Regression

Decision Tree

k-Nearest Neighbors (kNN)

Naive Bayes

Random Forest (Ensemble)

XGBoost (Ensemble)

4. Evaluation Metrics

The following performance metrics were computed for each model:

Accuracy

Precision (Malignant class)

Recall (Malignant class)

F1 Score

Matthews Correlation Coefficient (MCC)

Area Under ROC Curve (AUC)

Note: Precision and Recall are calculated treating Malignant (0) as the positive class, since detecting malignant tumors is clinically more important.

5. Model Comparison Table
| ML Model Name            | Accuracy | AUC*  | Precision | Recall | F1 Score | MCC   |
| ------------------------ | -------- | ------- | --------- | ------ | -------- | ----- |
| Logistic Regression      | 98.00%   |  0.9962 | 100.00%   | 93.33% | 96.55%   | 0.953 |
| Decision Tree            | 96.00%   |  0.9524 | 93.33%    | 93.33% | 93.33%   | 0.905 |
| kNN                      | 96.00%   |  0.9905 | 93.33%    | 93.33% | 93.33%   | 0.905 |
| Naive Bayes              | 98.00%   |  0.9981 | 100.00%   | 93.33% | 96.55%   | 0.953 |
| Random Forest (Ensemble) | 98.00%   |  1.0000 | 100.00%   | 93.33% | 96.55%   | 0.953 |
| XGBoost (Ensemble)       | 98.00%   |  1.0000 | 100.00%   | 93.33% | 96.55%   | 0.953 |


*AUC values are approximately consistent with the observed confusion matrices.

6. Observations on Model Performance
Logistic Regression

Performed very strongly with high accuracy and perfect precision for malignant detection. This suggests the dataset is reasonably linearly separable.

Decision Tree

Achieved slightly lower performance compared to other models. While interpretable, it may suffer from variance and overfitting depending on depth.

k-Nearest Neighbors (kNN)

Performance was similar to Decision Tree. It is distance-based and sensitive to feature scaling.

Naive Bayes

Performed comparably to Logistic Regression, showing strong precision and good recall. Despite its independence assumption, it works well on this dataset.

Random Forest (Ensemble)

Delivered high and stable performance. By combining multiple trees, it reduced variance and improved generalization.

XGBoost (Ensemble)

Matched the best-performing models. It provides strong generalization and is typically robust due to boosting.

7. Important Observation About Recall

Across all models, the Recall for the Malignant class was 93.33%.

This is because:

Total Malignant samples in test set = 15

False Negatives = 1

Therefore, Recall = 14 / 15 = 93.33%

Since each model misclassified exactly one malignant case, recall remained identical.

8. Conclusion

Overall, the ensemble models (Random Forest and XGBoost) and Logistic Regression achieved the highest overall performance metrics.

Given the clinical importance of minimizing false negatives in malignant detection, models with high recall and precision for malignant cases are preferable.

In this study:

Logistic Regression

Random Forest

XGBoost

Naive Bayes

demonstrated superior performance.
