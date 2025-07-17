from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib


model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


feature_list = joblib.load("model_features.pkl")  


loan_model = joblib.load("catboost_loan_model_regularized_fixed.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/getstarted')
def get_started():
    return render_template("getstarted.html")

@app.route('/churn')
def churn_page():
    return render_template("churn.html")

@app.route('/loan')
def loan_page():
    return render_template("loan.html")
@app.route('/predict_by_input', methods=["POST"])
def predict_by_input():
    try:
        # Get form inputs
        gender = request.form['gender']
        geography = request.form['geography']
        marital_status = request.form['marital_status']
        age = float(request.form['age'])
        tenure = float(request.form['tenure'])
        balance = float(request.form['balance'])
        num_of_products = float(request.form['num_of_products'])
        estimated_salary = float(request.form['estimated_salary'])
        credit_score = float(request.form['credit_score'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])

        # Engineering
        monthly_income = estimated_salary
        emi_amount = balance / 12 if balance > 0 else 0
        sentiment_score = 0.5  

        # Combine categorical data as trained
        combined_cats = f"{gender} {geography} Chennai {marital_status} Savings Yes 101 Happy"
        tfidf_features = tfidf.transform([combined_cats])
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf.get_feature_names_out())

        # Numeric Data
        numeric_data = [[
            age, tenure, balance, num_of_products, estimated_salary,
            monthly_income, credit_score, emi_amount, sentiment_score
        ]]
        numeric_df = pd.DataFrame(scaler.transform(numeric_data), columns=[
            'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',
            'Monthly_Income', 'CreditScore', 'EMI_Amount', 'Sentiment_Score'
        ])

        # Combine numeric + TF-IDF features
        final_input = pd.concat([numeric_df, tfidf_df], axis=1)

        # Ensure all model features exist
        for col in feature_list:
            if col not in final_input:
                final_input[col] = 0
        final_input = final_input[feature_list]

        # Predict
        prediction = model.predict(final_input)[0]
        label = " Customer is likely to churn" if prediction == 1 else " Customer is likely to stay"

        # SHAP explanation (list only, no plot)
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(final_input)
        feature_names = final_input.columns.tolist()
        shap_tuples = sorted(
            zip(feature_names, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_factors = shap_tuples[:5]

        return render_template(
            "result.html",
            result=label,
            top_factors=top_factors
        )

    except Exception as e:
        return f"Error during prediction: {e}"

@app.route('/predict_loan', methods=["POST"])
def predict_loan():
    try:
        # Get form inputs from loan.html
        Gender = request.form['gender']
        Married = request.form['married']
        Dependents = request.form['dependents']
        Education = request.form['education']
        Self_Employed = request.form['self_employed']
        ApplicantIncome = float(request.form['applicant_income'])
        CoapplicantIncome = float(request.form.get('coapplicant_income', 0) or 0)
        LoanAmount = float(request.form.get('loan_amount', 0) or 0)
        Loan_Amount_Term = float(request.form['loan_amount_term'])
        Credit_History = float(request.form['credit_history'])
        Property_Area = request.form['property_area']

        # Feature engineering (must match training)
        Dependents_num = 3 if Dependents == '3+' else int(Dependents)
        Total_Income = ApplicantIncome + CoapplicantIncome
        Income_per_Member = Total_Income / (Dependents_num + 1)
        Loan_Amount_to_Income = LoanAmount / (Total_Income + 1)
        EMI = LoanAmount / Loan_Amount_Term if Loan_Amount_Term > 0 else 0
        Balance_Income = Total_Income - (EMI * 1000)
        EMI_ratio = EMI / (Total_Income + 1)
        Loan_Income_log = np.log1p(LoanAmount / (Total_Income + 1))
        Credit_History_Effect = Credit_History * Loan_Amount_to_Income
        LoanAmount_Education = LoanAmount * (1 if Education == 'Graduate' else 0)
        Has_Coapplicant = 1 if CoapplicantIncome > 0 else 0
        Income_Term_Ratio = Total_Income / (Loan_Amount_Term + 1)

        # Prepare input DataFrame
        input_dict = {
            'Gender': Gender,
            'Married': Married,
            'Dependents': Dependents_num,
            'Education': Education,
            'Self_Employed': Self_Employed,
            'ApplicantIncome': ApplicantIncome,
            'CoapplicantIncome': CoapplicantIncome,
            'LoanAmount': LoanAmount,
            'Loan_Amount_Term': Loan_Amount_Term,
            'Credit_History': Credit_History,
            'Property_Area': Property_Area,
            'Total_Income': Total_Income,
            'Income_per_Member': Income_per_Member,
            'Loan_Amount_to_Income': Loan_Amount_to_Income,
            'EMI': EMI,
            'Balance_Income': Balance_Income,
            'EMI_ratio': EMI_ratio,
            'Loan_Income_log': Loan_Income_log,
            'Credit_History_Effect': Credit_History_Effect,
            'LoanAmount_Education': LoanAmount_Education,
            'Has_Coapplicant': Has_Coapplicant,
            'Income_Term_Ratio': Income_Term_Ratio
        }
        input_df = pd.DataFrame([input_dict])

        # Debug: Print input and model output
        print("Loan input features:")
        print(input_df)

        # Predict probability and adjust threshold (e.g., 0.6 for stricter approval)
        proba = loan_model.predict_proba(input_df)[0][1]
        print(f"Predicted approval probability: {proba:.4f}")
        threshold = 0.5  
        pred = 1 if proba > threshold else 0
        label = f"{'✅ Loan Approved' if pred == 1 else '❌ Loan Not Approved'} (Probability: {proba:.2%})"

        # Feature importance for this prediction (using CatBoost's prediction values)
        import shap
        explainer = shap.TreeExplainer(loan_model)
        shap_values = explainer.shap_values(input_df)
        feature_names = input_df.columns.tolist()
        shap_tuples = sorted(
            zip(feature_names, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_factors = shap_tuples[:5]

        return render_template("resultloan.html", result=label, top_factors=top_factors)
    except Exception as e:
        print(f"Error during loan prediction: {e}")
        return f"Error during loan prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)