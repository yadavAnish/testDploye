#import libraries
import io
import base64
import os


import random

from os import urandom
from flask.helpers import flash, send_file
from flask.wrappers import Response
from matplotlib import figure
import pandas as pd

from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template,request
import pickle#Initialize the flask App
import seaborn as sns
from matplotlib.figure import Figure
plt.style.use('seaborn-bright')


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))




#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    print(output)
    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS NOT LIKELY TO HAVE A HEART FAILURE'.format(output))
    else:
         return render_template('index.html', prediction_text='THE PATIENT IS LIKELY TO HAVE A HEART FAILURE')




picFolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picFolder

@app.route("/visual",methods=['POST'])
def visual():
    imageList = os.listdir('static/pics')
    imagelist = ['pics/' + image for image in imageList]
    return render_template("dashboard.html", imagelist=imagelist)




if __name__ == "__main__":
    app.run(debug=True)


    