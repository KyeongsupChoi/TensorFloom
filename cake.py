import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
tf.disable_eager_execution()

m = hub.Module('https://tfhub.dev/google/aiy/vision/classifier/food_V1/1')

import numpy as np
import pandas as pd
import cv2
from skimage import io


cake_url = "https://stylesweet.com/wp-content/uploads/2022/06/ChocolateCakeforTwo_01.jpg"
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