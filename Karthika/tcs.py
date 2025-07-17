import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import emoji
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# ======================= 1. Load Dataset ===========================
filename = os.path.join(os.getcwd(), 'sentiment_analysis.csv')
df = pd.read_csv(filename)

print("Initial Info:")
print(df.info())
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

# ======================= 2. Visualizations =========================
sns.countplot(data=df, x='Exited')
plt.title("Churn Distribution")
plt.show()

sns.histplot(df['Sentiment_Score'], bins=20, kde=True)
plt.title("Distribution of Sentiment Scores")
plt.show()

# ======================= 3. Feature Engineering ====================
df.drop(['CustomerID', 'Name', 'Account_Open_Date', 'Last_Transaction_Date'], axis=1, inplace=True)

cat_cols = ['Gender', 'Geography', 'Native_Place', 'Marital_Status',
            'Account_Type', 'Loan_Status', 'Branch_Code', 'Review_Sentiment']
df[cat_cols] = df[cat_cols].astype(str)

# Combine categorical columns into a single string
df['combined_cats'] = df[cat_cols].agg(' '.join, axis=1)

# TF-IDF on combined categorical string
tfidf = TfidfVectorizer(max_features=500)
cat_tfidf = tfidf.fit_transform(df['combined_cats'])
cat_tfidf_df = pd.DataFrame(cat_tfidf.toarray(), columns=tfidf.get_feature_names_out(), index=df.index)

# Drop original categorical columns
df.drop(columns=cat_cols + ['combined_cats'], inplace=True)

# Store target variable and drop it temporarily from df for processing
y = df['Exited']
X = df.drop(columns=['Exited', 'Customer_Review'])  # assuming 'Customer_Review' is unused

# Scale numeric features
cols_to_scale = ['Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
                 'Monthly_Income', 'CreditScore', 'EMI_Amount', 'Sentiment_Score']
scaler = StandardScaler()
X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

# Combine numeric + tfidf features
X_final = pd.concat([X, cat_tfidf_df], axis=1)
print("Final dataframe shape:", X_final.shape)
print(X_final.head())

# ======================= 4. Train-Test Split =======================
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, stratify=y, random_state=42
)

# ======================= 5. XGBoost Training =======================
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy:  {test_acc:.4f}")

# ======================= 6. Accuracy Plot ==========================
plt.figure(figsize=(6, 4))
plt.bar(['Training Accuracy', 'Testing Accuracy'], [train_acc, test_acc], color=['mediumseagreen', 'steelblue'])
plt.ylim(0, 1)
plt.title('XGBoost Model Accuracy')
plt.ylabel('Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ======================= 7. Save Models ============================
joblib.dump(xgb_model, "xgb_churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(X_final.columns.tolist(), "model_features.pkl")  # Save feature order
print("Models and feature names saved successfully.")

# ======================= 8. Sample Predictions =====================
model = joblib.load("xgb_churn_model.pkl")
sample_input = X_test.iloc[:5]
prediction = model.predict(sample_input)

print("\nSample Predictions:")
for i in range(5):
    print(f"Customer {i+1}:")
    print("Prediction:", prediction[i])
    print("True Label:", y_test.iloc[i])
    print("✅ Correct\n" if prediction[i] == y_test.iloc[i] else "❌ Incorrect\n")

# Final test accuracy
final_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Final Test Accuracy: {final_acc:.4f}")
import joblib

feature_list = joblib.load("model_features.pkl")
print("Model was trained on features in this order:\n")
for i, col in enumerate(feature_list, 1):
    print(f"{i}. {col}")
