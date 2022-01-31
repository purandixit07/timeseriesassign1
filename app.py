# importing the necessary dependencies
# from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.arima_model import ARIMA
import pickle

import numpy as np
from flask import Flask, render_template, request
from flask_cors import cross_origin

app = Flask(__name__)  # initializing a flask app


@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            steps = float(request.form['steps'])
            filename = 'modelForPrediction.sav'

            loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            # predictions using the loaded model file

            prediction = loaded_model.forecast(steps=steps)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            result = np.exp(prediction[0])
            render_template('results.html', result=result)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
#if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080)
    # port = int(os.getenv("PORT"))
    # host = '0.0.0.0'
    # httpd = simple_server.make_server(host=host,port=5000, app=app)
    # print("localhost:5000")
    # httpd.serve_forever()
