from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent.parent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


labels = ["Normal", "Stone"]
base_dir = os.path.join(BASE_DIR, 'static/data/')

train_path = base_dir + 'train/'
test_path = base_dir + 'test/'
valid_path = base_dir + 'val/'

batch_size = 16 
img_height = 500
img_width = 500

from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(
                                  rescale = 1./255,
                                  shear_range = 0.2,
                                  zoom_range = 0.2,
                                  horizontal_flip = True,          
                               )

test_data_gen = ImageDataGenerator(rescale = 1./255)


train = image_gen.flow_from_directory(
      train_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary',
      batch_size=batch_size
      )
test = test_data_gen.flow_from_directory(
      test_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      shuffle=False, 
      class_mode='binary',
      batch_size=batch_size
      )

valid = test_data_gen.flow_from_directory(
      valid_path,
      target_size=(img_height, img_width),
      color_mode='grayscale',
      class_mode='binary', 
      batch_size=batch_size
      )


plt.figure(figsize=(12, 12))
for i in range(0, 10):
    plt.subplot(2, 5, i+1)
    for X_batch, Y_batch in train:
        image = X_batch[0]        
        dic = {0:’NORMAL’, 1:’PNEUMONIA’}
        plt.title(dic.get(Y_batch[0]))
        plt.axis(’off’)
        plt.imshow(np.squeeze(image),cmap=’gray’,interpolation=’nearest’)
        break
plt.tight_layout()
plt.show()