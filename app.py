from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd

app = Flask(__name__)
cv=TfidfVectorizer(max_features = 100)
loaded_model = pickle.load(open('model4.pkl', 'rb'))



def fake_jobs_det(jobs):
    input_data = [jobs]
    vectorized_input_data = cv.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_jobs_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)

