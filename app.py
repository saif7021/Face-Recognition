from __future__ import division, print_function
import os
import cv2
import glob
import re
import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask,url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from skimage.transform import resize
import cv2

app = Flask(__name__)

MODEL_PATH = 'face recognition128.h5'




def model_predict(img_path, model):
    #img = image.load_img(img_path, target_size=(64,64))
    img = cv2.imread(img_path)
    #print(type(img))
    try:
        img = resize(img,(128,128))
        print("after resize")
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1): 
            img = img/255.0
        prediction = model.predict(img)
        return prediction
    except AttributeError:
        return "shape not found"



@app.route('/', methods=['GET'])
def index():
    return render_template('base.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
       f= request.files['image']

       basepath = os.path.dirname(__file__)
       file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
       f.save(file_path)
       model = load_model('face recognition128.h5')
       preds = model_predict(file_path, model)
       print("123")
       data1=list(preds[0])
       a=max(data1)
       b=data1.index(a)
       print(b)
       if b==0:
            result = "This is Hrithik Roshan"
       if b==1:
            result = "This is Shahrukh Khan"
       if b==2:
            result = "This is Tiger Shroff"
       return result
       

if __name__ == '__main__':

    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
