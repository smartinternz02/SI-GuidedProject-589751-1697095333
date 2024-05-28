from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('naive_bayes_model.pkl')  # Use joblib to load the model

# Load your machine learning model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        size = float(request.form['size'])
        fuel = float(request.form['fuel'])
        distance = float(request.form['distance'])
        decibel = float(request.form['decibel'])
        airflow = float(request.form['airflow'])
        frequency = float(request.form['frequency'])

        total = np.array([[size, fuel, distance, decibel, airflow, frequency]])
        y_test = model.predict(total)

        if y_test[0] == 0:
            result = "The fire is in the extension state"
        else:
            result = "The fire is in non extenction state"

        return render_template('home.html', result=result)

    except Exception as e:
        return render_template('home.html', result="invalid input")

@app.route("/result", methods=['GET'])  # Handle GET requests for /result
def show_result():
    return render_template('home.html', result="")

if __name__ == "__main__":
    app.run(debug=True)
