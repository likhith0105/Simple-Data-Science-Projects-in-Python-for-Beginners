from flask import Flask, render_template_string, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///covid.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Model
class CovidData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20))
    confirmed = db.Column(db.Integer)
    recovered = db.Column(db.Integer)
    deaths = db.Column(db.Integer)

with app.app_context():
    db.create_all()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>COVID-19 Trend Analysis</title>
</head>
<body>
    <h1>COVID-19 Trend Analysis & Forecasting</h1>

    <form method="POST">
        <input type="text" name="date" placeholder="Date">
        <input type="number" name="confirmed" placeholder="Confirmed Cases">
        <input type="number" name="recovered" placeholder="Recovered Cases">
        <input type="number" name="deaths" placeholder="Deaths">
        <button type="submit">Add Data</button>
    </form>

    <h2>COVID Data</h2>
    <table border="1">
        <tr>
            <th>Date</th>
            <th>Confirmed</th>
            <th>Recovered</th>
            <th>Deaths</th>
        </tr>
        {% for row in data %}
        <tr>
            <td>{{ row.date }}</td>
            <td>{{ row.confirmed }}</td>
            <td>{{ row.recovered }}</td>
            <td>{{ row.deaths }}</td>
        </tr>
        {% endfor %}
    </table>

    <h2>Forecast</h2>
    <p>Predicted Next Day Cases: {{ prediction }}</p>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":
        record = CovidData(
            date=request.form["date"],
            confirmed=int(request.form["confirmed"]),
            recovered=int(request.form["recovered"]),
            deaths=int(request.form["deaths"])
        )
        db.session.add(record)
        db.session.commit()

    data = CovidData.query.all()

    prediction = "Not enough data"

    if len(data) >= 2:
        cases = [d.confirmed for d in data]

        X = np.array(range(len(cases))).reshape(-1, 1)
        y = np.array(cases)

        model = LinearRegression()
        model.fit(X, y)

        next_day = np.array([[len(cases)]])
        prediction = int(model.predict(next_day)[0])

    return render_template_string(
        HTML,
        data=data,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)