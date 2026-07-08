### Credit Card Fraud Detection

An end-to-end machine learning project that detects fraudulent credit card transactions using a Logistic Regression model, served via a FastAPI web service and deployed on Render.


Table of Contents


1. Problem Statement
2. Dataset
3. Project Structure
4. Tech Stack
5. Software And Tools Requirements
6. Model Workflow
7. Installation
8. Running Locally
9. API Usage
10. Deployment (Render)
11. Results
12. Limitations & Future Improvements

Dataset

1. Source: creditcard.csv (Kaggle — Credit Card Fraud Detection dataset)

2. Features: Time, V1–V28 (PCA-anonymized transaction features), Amount

3. Target: Class (0 = legitimate, 1 = fraudulent)

4. Class imbalance: Fraud cases are a very small percentage of the full dataset (492 fraud transactions in the original Kaggle version)

5. Balancing strategy: Random undersampling — legitimate transactions are downsampled to match the number of fraud cases, creating a balanced training set


### Tech Stack

1. Language: Python 3.x
2. ML Libraries: scikit-learn, pandas, numpy
3. Visualization: seaborn
4. Web Framework: FastAPI + Pydantic
5. Model Serialization: joblib
6. Deployment: Render

### Software And Tools Requirements

1. [GitHub Account](https://github.com)
2. [Render Account](https://)
3. [VS Code IDE](https://code.visualstudio)
4. Git
5. Python 3.8+