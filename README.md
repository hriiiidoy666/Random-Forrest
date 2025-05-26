# Crop Classification Using Machine Learning

## Overview

This project addresses a **multi-class classification problem** using nutrient data extracted from various crops. The goal is to accurately classify crops based on the average amount of nutrients/elements found in them. The dataset provided (`Crop.csv`) contains both feature data and a target variable labeled as `Label`, representing the type of crop.

The crops under consideration include:

* Pomegranate
* Mango
* Grapes
* Mulberry
* Ragi
* Potato

This project follows a complete machine learning workflow including preprocessing, model selection, parameter tuning, evaluation, and visualization.

---

## Dataset Description

* **File Name**: `Crop.csv`
* **Features**: Numeric values representing the average concentration of nutrients/elements.
* **Target Variable**: `Label` (categorical, representing crop types)

---

## Problem Statement

Develop a machine learning classification model that:

1. Loads and preprocesses the dataset.
2. Scales all feature values to a standard range \[0, 1].
3. Trains a classification model using the processed data.
4. Applies hyperparameter tuning and cross-validation to improve accuracy.
5. Evaluates the model using precision, recall, and F1-score.
6. Visualizes the results using a confusion matrix.

---

## Methodology

### 1. Data Loading

* Imported the dataset using `pandas`.
* Displayed basic statistics and verified data types and missing values.

### 2. Data Preprocessing

* Applied **Min-Max Scaling** to normalize all feature values between 0 and 1.
* Performed label encoding if necessary.

### 3. Model Selection

* Chose an appropriate classification algorithm (e.g., **Random Forest**, **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**).
* Split the dataset into training and test sets using **train-test split**.

### 4. Hyperparameter Tuning & Cross-Validation

* Used **GridSearchCV** or **RandomizedSearchCV** for parameter tuning.
* Applied **k-fold cross-validation** to ensure generalizability and reduce overfitting.

### 5. Evaluation Metrics

* Calculated the following metrics:

  * **Precision**
  * **Recall**
  * **F1-Score**
* Generated and plotted a **confusion matrix** to visualize class-wise performance.

### 6. Final Output

* Displayed the classification report.
* Visualized the confusion matrix using `seaborn`.

---

## Technologies Used

| Tool/Library   | Purpose                          |
| -------------- | -------------------------------- |
| `Python`       | Programming language             |
| `pandas`       | Data loading and manipulation    |
| `numpy`        | Numerical operations             |
| `scikit-learn` | Model building, scaling, metrics |
| `matplotlib`   | Plotting                         |
| `seaborn`      | Enhanced visualization           |

---

## File Submission

  📄 `Q2_Hridoy.ipynb`


---

## Author

**Name**: Hridoy Hossain     
**Institution**: Khulna University    
**Course**:  Python Programming and Data Science Basics
