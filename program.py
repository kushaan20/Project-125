import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression 
from PIL import Image
import PIL.ImageOps 

X = np.load('image.npz')['arr_0']
y = pd.read_csv()("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
x_train, x_test, y_train, y_test = train_test_split(X,y,random_state = 9, train_size = 7500, test_size = 2500)

x_train_scaled = x_train/255
x_test_scaled = x_test/255

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_train_scaled, y_train)

def get_prediction(image):
    image_pil = Image.open(image)
    image_bw = image_pil.convert('L')
    image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resize,pixel_filter)
    image_bw_resize_inverted_scale = np.click(image_bw_resize - min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resize)
    image_bw_resize_inverted_scale = np.asarray(image_bw_resize_inverted_scale)/max_pixel
    test_sample = np.array(image_bw_resize_inverted_scale).reshape(1,784)
    test_prediction = clf.predict(test_sample)
    return test_prediction[0]

from flask import Flask, jsonify, request
from classifier import get_prediction 
app = Flask(__name__)
@app.route("/predict-digit", methods = ["POST"])

def predict_data():
    image = request.files.get('digit')
    prediction = get_prediction(image)
    return jsonify({
        "prediction": prediction
    }),200

if __name__ == "__main__":
    app.run(debug = True)