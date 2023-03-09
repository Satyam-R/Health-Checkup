from flask import Flask, render_template, url_for, flash, redirect
import pickle
import joblib
from flask import request
import numpy as np

app = Flask(__name__, template_folder='templatefiles', static_folder='assets')


@app.route("/")
def index() :
    return render_template("index.html")

@app.route("/liver")
def cancer():
    return render_template("liver_index.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if (size == 7):
        loaded_model = joblib.load('liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predict1', methods=["POST"])
def predict1():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        # liver
        if (len(to_predict_list) == 7):
            result = ValuePredictor(to_predict_list, 7)

    if (int(result) == 1):
        prediction = "Sorry you chances of getting the disease. Please consult the doctor immediately"
    else:
        prediction = "No need to fear. You have no dangerous symptoms of the disease"
    return (render_template("liver_res.html", prediction_text=prediction))


### Diabetes

filename = 'diabetes_model.pkl'
classifier = pickle.load(open(filename, 'rb'))


@app.route('/diabetes')
def home():
    return render_template('Diabetes_index.html')


@app.route('/predict2', methods=['POST'])
def predict2():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('Diabetes_res.html', prediction=my_prediction)


if __name__ == "__main__":
    app.run(debug=True)
