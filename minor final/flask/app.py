from flask import Flask, request, render_template
from joblib import load
import numpy as np
import os

app = Flask(__name__)
model_path = 'best_decision_tree_model.pkl'

# Debugging: Print the current working directory and list files
print("Current working directory:", os.getcwd())
print("Files in the current directory:", os.listdir(os.getcwd()))

# Check if the model file exists before loading
if os.path.exists(model_path):
    model = load(model_path)
    print("Model loaded successfully.")
else:
    model = None
    print(f"Model file not found: {model_path}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if model is None:
        return render_template('home.html', result='Fire extinction state.')

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
            results = "The fire is in extension state"
        else:
            results = "The fire is in non-extension state"

        return render_template('home.html', result=results)

    except Exception as e:
        print(e)  # Log the specific error for debugging
        return render_template('home.html', result='Invalid input')

if __name__ == "__main__":
    app.run(debug=True)
