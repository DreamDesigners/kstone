
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
base_dir = os.path.join(BASE_DIR, 'static/data/')
#test_image_path = base_dir + 'test_1.jpg'

import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set_style("whitegrid")
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img

def test(test_image=None):
        print("testing")
        model = tf.keras.models.load_model('model.h5')
        model.summary()


        # pridict if test_image_path is a stone or normal

        # load image
        img = load_img(test_image, target_size=(150, 150))
        img_array = np.array(img)
        img_array = img_array/255
        img_array = img_array.reshape(1, 150, 150, 3)


        # predict
        prediction = model.predict(img_array)
        print(prediction)


        # print accuracy


        # print prediction
        if prediction[0][0] > 0.5:
                print("Stone")
        else:
                print("Normal")

        return prediction[0][0]


#test(test_image_path)