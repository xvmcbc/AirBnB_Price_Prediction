import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from flask import (
    Flask,
    render_template,
    jsonify,
    request)

app = Flask(__name__)


@app.route('/')
def home():

    var = 2
    return render_template('index.html', my_var = var)

@app.route('/map1')
def map1():
    return render_template('map1.html')

@app.route('/map2')
def map2():
    return render_template('map2.html')

my_data =[]

#Modelo de Regresion
@app.route('/model', methods=["GET", "POST"])
def model():

    #Importa el modelo entrenado del archivo
    modelo_regresion = pickle.load(open('Regressor_model_cut.sav', 'rb'))

    #Creacion dataframe para variables (el archivo tiene que estar en la ruta de app.py)
    data_X = pd.read_csv('X_temp.csv')

    #Asignación de valores desde la forma
    if request.method == "POST":
        accommodates = request.form["accommodates"]
        tv = request.form["tv"]
        pool = request.form["pool"]
        room_type = request.form["room_type"]
        cancellation_policy = request.form["cancellation_policy"]
        neighbourhood = request.form["neighbourhood"]
        
        #Features adjustment
        data_X['accommodates'] = data_X['accommodates'].replace(0, accommodates)
        data_X['Cable TV/TV'] = data_X['Cable TV/TV'].replace(0, tv)
        data_X['Hot tub/jetted tub/private hot tub/sauna/shared hot tub/pool/private pool/shared pool'] = data_X['Hot tub/jetted tub/private hot tub/sauna/shared hot tub/pool/private pool/shared pool'].replace(0, pool)    

        data_X[room_type] = data_X[room_type].replace(0, 1)
        data_X[neighbourhood] = data_X[neighbourhood].replace(0, 1)
        data_X[cancellation_policy] = data_X[cancellation_policy].replace(0, 1)

        #Extracting only numerical data
        numerical_columns = ['accommodates']

        #Log transforming columns
        for col in numerical_columns:
            data_X[col] = data_X[col].astype('float64').replace(0.0, 0.01) # Replacing 0s with 0.01
            data_X[col] = np.log(data_X[col])

        # Scaling Data------------------------
        #Using scaler from the model already trained
        scaler = joblib.load('scaler.pkl') 

        X = np.array(data_X)
        data_X = scaler.transform(X.reshape(1, -1))

        #Predicting...
        y_load_predit = modelo_regresion.predict(data_X)

        #Transforming again
        prediccion = np.exp(y_load_predit)
        prediccion = str(prediccion)

        #return f" El precio según el modelo será alrededor de {prediccion} MXN"
        prediccion = prediccion.replace("[","").replace("]","")
        return render_template("result.html", var=prediccion)

    return render_template("model.html")

if __name__ == '__main__':
    app.run()