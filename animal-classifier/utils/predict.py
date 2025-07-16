from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_image
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'animal_classifier.h5')
model = load_model(MODEL_PATH)

class_names = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin',
               'Elephant', 'Giraffe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

def classify_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)[0]
    return class_names[np.argmax(predictions)]
