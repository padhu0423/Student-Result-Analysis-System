from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    maths = int(request.form['maths'])
    science = int(request.form['science'])
    english = int(request.form['english'])
    attendance = int(request.form['attendance'])

    total = maths + science + english
    average = total / 3

    prediction = model.predict([[maths, science, english, attendance]])

    return render_template(
        'result.html',
        total=total,
        average=average,
        result=prediction[0]
    )

if __name__ == '__main__':
    app.run(debug=True)
