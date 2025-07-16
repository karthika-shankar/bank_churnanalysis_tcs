# CrediPlus - Bank Churn Analysis and Loan Prediction 


## Problem Statement
Financial institutions face two major risks that significantly impact their long-term profitability and sustainability: customer churn and loan defaults. Customer churn leads
to a decline in customer lifetime value and increases customer acquisition costs, making
retention a critical focus area. Loan defaults, on the other hand, result in direct financial
losses and negatively affect the institution’s credit stability.
A machine learning–driven solution is needed to predict these risks early using structured and unstructured data, enabling businesses to proactively reduce losses and improve
customer retention

## Objectives
The primary objective of this project is to leverage machine learning to address critical
business challenges in the banking domain, starting with customer churn prediction. The
project aims to predict customer churn by analyzing demographic, transactional, and
sentiment-related features, thereby enabling proactive customer retention strategies. In
addition, it seeks to predict loan default based on customer profiles and financial behavior
to support better credit risk assessment. The solution also focuses on identifying key risk
drivers through explainable machine learning methods, providing actionable insights for
departments such as HR, customer care, and risk management. Lastly, the project aims
to build reusable machine learning pipelines that can be scaled and deployed effectively
in real-world production environments.

## Approach
To ensure robust performance and identify the most suitable algorithm for our bank
churn and loan risk prediction tasks, we experimented with a diverse range of machine
learning models. These included both classical and modern approaches such as Logistic
Regression, K-Nearest Neighbors (KNN), and Random Forest. We further evaluated
advanced ensemble methods like XGBoost, LightGBM, and CatBoost, known for their
high accuracy and ability to handle complex, high-dimensional datasets. Additionally, we
implemented a Neural Network using to explore deep learning’s capabilities in capturing
nonlinear relationships within the data. 
The dataset was initially preprocessed using TF-IDF, with the top 500 features retained
to represent categorical information numerically. Unstructured customer review text was processed separately using a pre-trained transformer model. Specifically, the RoBERTa base model from Hugging Face was used to
compute sentiment scores for each review.
Various preprocessing techniques like TF-IDF, One-Hot Encoding, and Label Encoding and feature engineering was carried out to achieve the desired results. The results were later analysed and compared using different performance metrics like ROC-AUC curve, Confusion matrix, Feature Importance chart, SHAP etc.
