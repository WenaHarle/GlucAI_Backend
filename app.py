from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        gender = request.form['gender']
        age = request.form['age']
        hypertension = request.form['hypertension']
        heart_disease = request.form['heart_disease']
        smoking_history = request.form['smoking_history']
        bmi = request.form['bmi']
        HbA1c_level = request.form['HbA1c_level']
        blood_glucose_level = request.form['blood_glucose_level']

        model = joblib.load('trained_model.joblib')


        new_instance = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })

        # Make the prediction
        prediction = model.predict(new_instance)

        return f"Prediction: {prediction[0]}"
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
