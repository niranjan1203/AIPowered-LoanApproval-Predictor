from flask import Flask, render_template, request
import pandas as pd
import joblib

model = joblib.load("loan_approval_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form.get('age', 0))
        income = float(request.form.get('income', 0))
        loan_amount = float(request.form.get('loan_amount', 0))
        credit_score = float(request.form.get('credit_score', 0))

        employment_type = request.form.get('employment_type', '').strip() 

        if employment_type not in ["0", "1"]:
            return render_template('index.html', prediction="Error: Employment Type is missing or invalid!")

        employment_type = int(employment_type)

        input_data = pd.DataFrame([[age, income, loan_amount, credit_score, employment_type]],
                                  columns=['Age', 'Income', 'Loan_Amount', 'Credit_Score', 'Employment_Type'])

        prediction = model.predict(input_data)

        result = "Approved" if prediction[0] == 1 else "Rejected"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

