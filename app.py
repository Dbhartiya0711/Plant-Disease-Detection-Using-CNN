import io
from tkinter import Image
import keras
import numpy as np
import tensorflow as tf
import imageio
from flask import Flask, request, jsonify, render_template
import PIL.Image
import pickle
import joblib
import cv2
from keras.utils.image_utils import img_to_array

# Create flask app
flask_app = Flask(__name__)

tf.saved_model.LoadOptions(
    experimental_io_device=None
)
model = keras.models.load_model("model_final.h5")
# model = keras.models.load_model("model_new.h5")
@flask_app.route("/api", methods=["GET"])
def Home():
    res = {"result": "Result"}
    return res

@flask_app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image']
    print(file)
    img = imageio.imread(file)
    img = cv2.resize(img, tuple((128, 128)))
    img = img[np.newaxis, :, :, :]
    # image_labels = pickle.load(open('label_transform.pkl', 'rb'))
    image_labels = pickle.load(open('label_new.pkl', 'rb'))
    prediction = model.predict(img)
    result = image_labels.classes_[np.argmax(prediction, axis=1)][0]
    res = {"result": result}
    return res
    # res = {"result": "Resssssss"}
    # return res