from flask import Flask, render_template, request
import pickle
import numpy as np
import sklearn


def time_to_seconds(time_str):
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60


knn = pickle.load(open(".\static\models\model.pkl", "rb"))


def trafficPrediction(data):
    data = np.array(data).reshape(1, -1)
    # Get the prediction
    result = knn.predict(data)
    return result


app = Flask(__name__)

# Route untuk halaman utama


@app.route('/')
def index():
    return render_template('index.html')

# route 'form' set as 'form.html'


@app.route('/form')
def form():
    return render_template('form.html')

# Route untuk menangani submit form


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Mengambil data dari form
        day = float(request.form['day'])
        time = request.form['time']
        car_count = float(request.form['car'])
        bike_count = float(request.form['bike'])
        bus_count = float(request.form['bus'])
        truck_count = float(request.form['truck'])

        time_seconds = time_to_seconds(time)
        count = (car_count, bike_count, bus_count, truck_count)
        total = sum(count)
        Data_Testing = [[time_seconds, day, car_count,
                         bike_count, bus_count, truck_count, total]]
        y_pred = knn.predict(Data_Testing)
        if y_pred == 0:
            hasil = "Low"
        elif y_pred == 1:
            hasil = "Normal"
        elif y_pred == 2:
            hasil = "High"
        elif y_pred == 3:
            hasil = "Heavy"
        else:
            hasil == "Error"

        return render_template("output.html", prediction=hasil)


if __name__ == '__main__':
    app.run(debug=True)
