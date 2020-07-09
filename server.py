import numpy as np
from flask import Flask, request, render_template
import pickle

from fastai.vision import open_image
from fastai.basic_train import load_learner
import torch
from PIL import Image 

import os

cwd = os.getcwd()
path = Path()

application = Flask(__name__)

model = load_learner('model/', 'squeezenet_model.pkl')

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #labels = ['grizzly','black','teddy']

    file = request.files['file']

    #open file
    img = open_image(file)

    #Getting the prediction from the model
    prediction = model.predict(img)[0]

    #Render the result in the html template
    return render_template('index.html', prediction_text='Your Prediction :  {} '.format(prediction))