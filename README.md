# Logistic Regression Classifier – Breast Cancer Detection

## Overview
This project demonstrates a simple binary classification model using Logistic Regression.  
The goal is to classify breast cancer tumors as **Malignant (1)** or **Benign (0)** based on the Breast Cancer Wisconsin dataset.

Dataset Source:  
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

This notebook explains:
- How logistic regression works  
- How to train a classification model  
- How to evaluate it using multiple metrics  
- How to visualize performance (confusion matrix, ROC curve, threshold plots)

---

## Model Performance

The Logistic Regression model achieved strong results:

- Precision: **0.9750**
- Recall: **0.9286**
- F1-Score: **0.9512**
- ROC-AUC Score: **0.9960**

### Confusion Matrix
[[71 1]
[ 3 39]]

(71 true negatives, 39 true positives)

---

## Visualizations Included

### 1. Confusion Matrix Plot
A heatmap showing correct vs incorrect predictions.

### 2. Precision vs Threshold Plot
Shows how precision and recall change when adjusting the decision threshold (default = 0.5).

### 3. ROC Curve
The ROC curve shows how well the classifier separates the two classes.  
A curve closer to the top-left indicates better performance.

### 4. Optimal Threshold Confusion Matrix
A comparison between:
- Default threshold (0.5)
- Optimal threshold (based on precision/recall balancing)

---

## Logistic (Sigmoid) Function Explanation

Logistic Regression uses the **sigmoid function** to convert any real number into a probability between 0 and 1.

### Sigmoid Formula
S(z) = 1 / (1 + e^(-z))


Where:
- z = w1*x1 + w2*x2 + ... + b  
  (linear combination of features and learned weights)

### Decision Rule
- If S(z) >= 0.5 → Predict class **1 (Malignant)**
- If S(z) < 0.5 → Predict class **0 (Benign)**

---

## What You Learn
- Basics of Logistic Regression  
- How classification works in machine learning  
- How to evaluate a classifier  
- How threshold changes affect performance  
- Meaning of precision, recall, F1-score, AUC  
- How to interpret confusion matrices and ROC curves

---

## Requirements
- Python  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

Install requirements using:

"pip install pandas numpy matplotlib seaborn scikit-learn"



### Simple Explaination of Sigmoid function
Logistic regression is named for its core function, the sigmoid (or logistic) function. This function takes any real-valued number and "squashes" it into a value between 0 and 1.

The formula is:
                $$S(z) = \frac{1}{1 + e^{-z}}$$


1. z is the linear combination of your features and the model's learned coefficients (e.g., $z = w_1x_1 + w_2x_2 + ... + b$).
2. The output, $S(z)$, is interpreted as the probability that a given sample belongs to the positive class (in our case, 'Malignant').
3. If $S(z) \ge 0.5$, the model predicts class 1.
4. If $S(z) < 0.5$, the model predicts class 0.



---

## Conclusion
This project provides a simple but powerful demonstration of Logistic Regression for classification problems.  
It helps understand how probabilities, thresholds, and metrics affect the final classification results.

