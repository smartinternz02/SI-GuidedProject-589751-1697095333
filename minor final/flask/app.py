from flask import Flask, request, render_template
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('best_decision_tree_model.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        SIZE = float(request.form['size'])
        FUEL = float(request.form['fuel'])
        DISTANCE = float(request.form['distance'])
        DECIBLE = float(request.form['decibel'])
        AIRFLOW = float(request.form['airflow'])
        FREQUENCY = float(request.form['frequency'])

        total = np.array([[SIZE, FUEL, DISTANCE, DECIBLE, AIRFLOW, FREQUENCY]])
        y_test = model.predict(total)

        if y_test[0] == 0:
            result = "The fire is in extension state"
        else:
            result = "The fire is in non-extension state"

        return render_template('home.html', result=result)

    except Exception as e:
        print(e)  # Log the specific error for debugging
        return render_template('home.html', result='Invalid input')

if __name__ == "__main__":
    app.run(debug=True)
