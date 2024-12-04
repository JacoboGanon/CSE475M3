from flask import Flask, render_template, request
from waitress import serve
import joblib
import pandas as pd

# Load the model
model = joblib.load('./rmf_model.pkl')

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        capital = float(request.form['capital'])
        liabilities = float(request.form['liabilities'])
        assets_last_year = float(request.form['assets_last_year'])
        net_income = float(request.form['net_income'])
        equity = float(request.form['equity'])
        sales = float(request.form['sales'])
        interest = float(request.form['interest'])
        interest_bearing_debt = float(request.form['interest_bearing_debt'])
        gross_income = float(request.form['gross_income'])
        eps1 = float(request.form.get('eps1', 0))
        eps2 = float(request.form.get('eps2', 0))
        eps3 = float(request.form.get('eps3', 0))
        eps4 = float(request.form.get('eps4', 0))

        # Check if capital + liabilities <= 0
        if capital + liabilities <= 0:
            return render_template('predict.html', bankrupt_status="Error: Capital + Liabilities must be greater than 0")

        # Calculate EPS average if any of the EPS values are empty
        if not eps1 or not eps2 or not eps3 or not eps4:
            eps_avg = 0.22
        else:
            eps_avg = (float(eps1) + float(eps2) + float(eps3) + float(eps4)) / 4

        # Calculate assets this year
        assets_this_year = float(liabilities) + float(equity)

        # Calculate Assets growth rate
        assets_growth_rate = ((assets_this_year - float(assets_last_year)) / float(assets_last_year)) - 1

        # Calculate ROE (If equity is 0, set ROE to 0)
        roe = float(net_income) / float(equity) if float(equity) != 0 else 0

        # Calculate Z-Score
        ebit = (float(net_income) / 0.7) + float(interest)
        z_score = 1.2 * (float(capital) / assets_this_year) + 1.4 * (float(net_income) / assets_this_year) + 3.3 * (ebit / assets_this_year) + 0.6 * (float(equity) / float(liabilities)) + 0.999 * (float(sales) / assets_this_year)

        # Calculate interest bearing debt interest rate (if interest bearing debt is 0, set rate to 0)
        interest_bearing_debt_interest_rate = float(interest) / float(interest_bearing_debt) if float(interest_bearing_debt) != 0 else 0

        # Calculate borrowing dependency (if gross income is 0, set dependency to 1)
        borrowing_dependency = float(liabilities) / float(gross_income) if float(gross_income) != 0 else 1

        # Make a prediction with the machine learning algorithm
        features = {
            " Net Value Growth Rate": assets_growth_rate,
            " Net Income to Stockholder's Equity": roe,
            " Persistent EPS in the Last Four Seasons": eps_avg,
            "z_score": z_score,
            " Interest-bearing debt interest rate": interest_bearing_debt_interest_rate,
            " Borrowing dependency": borrowing_dependency
        }
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)
        if prediction[0] == 1:
            text_prediction = "Bankrupt"
        else:
            text_prediction = "Not Bankrupt"

        return render_template('predict.html', bankrupt_status=text_prediction)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=3000)
    # app.run(host='0.0.0.0', port=3000, debug=True)