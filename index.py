# Flask Web App
import numpy as np
import pickle
from flask import Flask, request, render_template

model = pickle.load(open('model_heart.pkl', 'rb'))  #Load ML model

app = Flask(__name__)   #Create environment with flask

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
    # Put all form entries values in a list 
    features = [float(i) for i in request.form.values()]
    # Convert features to array
    array_features = [np.array(features)]
    # Predict features
    prediction = model.predict(array_features)

    output = prediction

    # Check the output values and retrive the result with html tag based on the value
    if output == 1:
        return render_template('index.html', result = 'ECV: poco probable, ¡pero no bajes la guardia!')
    else:
        return render_template('index.html', result = 'ECV: probable. Consulta con tu especialista y déjanos tu dato de contacto.')

if __name__ == '__main__':
    app.run(debug=True)



