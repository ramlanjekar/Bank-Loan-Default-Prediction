# Bank Loan Default Predictions

## Project Overview
The **Loan Default Prediction (LDP)** project aims to predict whether a borrower will repay a loan based on their financial and demographic features. Given the large dataset of 1,00,000 rows and 2,000 columns, the challenge was to optimize the data for processing and build an efficient machine learning model.

## Data Preprocessing

### Data Optimization 
Due to the large dataset size,data optimization techniques were used to ensure smooth processing on a 16GB RAM system. A custom data optimization function optimizes memory usage in a pandas DataFrame by downcasting numerical columns (integers and floats) to smaller, more memory-efficient data types. It checks the minimum and maximum values of each column and converts the data type to the smallest possible type that can hold the values. This reduces the DataFrame's memory usage, which is especially useful when working with large datasets. 
The dataset was highly unbalanced, with a significant skew in the distribution of loan repayments, so special attention was given to address this issue.

### SMOTE-Tomek
To address the class imbalance, the **SMOTE-Tomek** method was applied. SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for the minority class, while Tomek links remove instances that are borderline and likely misclassified. This combination helps create a balanced dataset, improving model performance and accuracy.

### Yeo-Johnson Transformation
Outliers in the dataset were removed using the **Yeo-Johnson transformation**, which is a power transformation method that normalizes data while handling both positive and negative values. This step ensured the model’s performance was not influenced by extreme outliers.

## Model Selection
A pipeline was created to streamline the process of evaluating different machine learning models with various hyperparameters. The pipeline enabled efficient testing and comparison of multiple models, ensuring the best one was selected based on performance.

## Model Training and Hyperparameter Tuning
After selecting the best model, the training process began on the preprocessed data. **Optuna** was used for hyperparameter tuning to optimize the model’s performance by automatically searching for the best hyperparameters across different models.

## Predictions
Once the model was trained and tuned, predictions were made on the processed dataset, and results were evaluated to assess the model's ability to predict loan defaults accurately.

