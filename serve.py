import os
import numpy as np
import pandas as pd

from flask import Flask, redirect, url_for, request, render_template
from skimage import io
import skimage.transform as st

from fastai.vision.learner import load_learner
from fastai.vision.models import squeezenet1_0
from fastai.vision.image import open_image



app = Flask(__name__)

@app.route('/')
def entry_page():
    return render_template('index.html')

@app.route('/predict_object/', methods=['GET', 'POST'])
def render_message():
    #Loading CNN model
    saved_model = 'saved_models/tuned_model_fin.h5'
    model = load_model(saved_model)
    
    try:
        #Get image URL as input
        image_url = request.form['image_url']
        image = io.imread(image_url)
        
        #Apply same preprocessing used while training CNN model
        image_small = st.resize(image, (224,224,3))
        x = np.expand_dims(image_small.transpose(2, 0, 1), axis=0)
        
        #Call classify function to predict the image class using the loaded CNN model
        final,pred_class = classify(x, model)
        print(pred_class)
        print(final)
        
        #Store model prediction results to pass to the web page
        message = "Model prediction: {}".format(pred_class)
        print('Python module executed successfully')
        
    except Exception as e:
        #Store error to pass to the web page
        message = "Error encountered. Try another image. ErrorClass: {}, Argument: {} and Traceback details are: {}".format(e.__class__,e.args,e.__doc__)
        final = pd.DataFrame({'A': ['Error'], 'B': [0]})
        
    #Return the model results to the web page
    return render_template('index.html',
                            message=message,
                            data=final.round(decimals=2),
                            image_url=image_url)
