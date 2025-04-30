# Linear vs Logistic Regression

This project compares the performance of **Linear Regression** and **Logistic Regression** for binary classification on a small dataset. The goal is to understand the differences between the two models and how they perform for classification tasks.

## Key Concepts:

### 1. **Linear Regression**:
   - **Linear Regression** fits a line to the data to predict a continuous output. In this case, it is used for binary classification, but it might not perform perfectly because it doesn't inherently constrain its predictions to binary values (i.e., 0 or 1).
   - Predictions are transformed into binary outcomes by setting a threshold of 0.5.

### 2. **Logistic Regression**:
   - **Logistic Regression** is a classification algorithm that predicts probabilities for each class (between 0 and 1). The model outputs the probability of the positive class, and we can classify data based on a threshold (typically 0.5).

## Code Overview:

### 1. **Data Preprocessing**:
   - The dataset consists of 10 points with one feature `x` and one binary target variable `y`. 
   - The feature is reshaped to a 2D array to meet the input requirements of scikit-learn models.

### 2. **Linear Regression Model**:
   - The `LinearRegression` model from `sklearn` is trained on the dataset.
   - The predictions are obtained, and they are transformed into binary predictions by setting a threshold of 0.5.

### 3. **Logistic Regression Model**:
   - The `LogisticRegression` model from `sklearn` is also trained on the dataset.
   - The model predicts probabilities, and the class is assigned based on a threshold of 0.5.

### 4. **Model Evaluation**:
   - The accuracy of both models is compared using the `accuracy_score` from `sklearn.metrics`.

### 5. **Visualization**:
   - A plot is created to compare the predictions of the **Linear Regression** and **Logistic Regression** models across a range of feature values from -6 to 22.
   - The plot shows the predicted probabilities for both models, along with a threshold line at 0.5.

## Results:

The accuracy of both models is:

- **Linear Regression Accuracy**: `0.6`
- **Logistic Regression Accuracy**: `1.0`

The plot visualizes the decision boundaries of both models, demonstrating the difference between the continuous output of linear regression and the constrained probability output of logistic regression.

## Conclusion:

This project demonstrates the differences between **Linear Regression** and **Logistic Regression** for binary classification. While **Linear Regression** might produce values outside the [0, 1] range, **Logistic Regression** naturally outputs probabilities that can be used for classification. In this case, Logistic Regression performs significantly better, achieving perfect accuracy on the dataset.
