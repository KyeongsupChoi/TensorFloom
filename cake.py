import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, redirect
tf.disable_eager_execution()

m = hub.Module('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')

import numpy as np
import pandas as pd
import cv2
from skimage import io
app = Flask(__name__)




cake_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR5xfBJ6rcoxym-AI0Xpqm0GFfwB_qMWyRsKUdekGrP5tzBWk6Xz1zVCyMvfbOrYfzCcMM&usqp=CAU"
labelmap_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_food_V1_labelmap.csv"
input_shape = (192, 192)


image = np.asarray(io.imread(cake_url), dtype="float")
image = cv2.resize(image, dsize=input_shape, interpolation=cv2.INTER_CUBIC)
# Scale values to [0, 1].
image = image / image.max()
# The model expects an input of (?, 224, 224, 3).
images = np.expand_dims(image, 0)
# This assumes you're using TF2.
output = m(images)

predicted_index = output.eval(session=tf.compat.v1.Session()).argmax()
classes = list(pd.read_csv(labelmap_url)["name"])
print("Prediction: ", classes[predicted_index])

stringy="Prediction: " + str(classes[predicted_index])

@app.route("/", methods=['GET', 'POST'])
def index():
    return stringy