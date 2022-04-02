from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent

import os
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


base_dir = os.path.join(BASE_DIR, 'static/data/')
train_dir = base_dir + 'train/'
test_dir = base_dir + 'test/'
val_dir = base_dir + 'val/'

print("Train set:")
print("-"*60)
num_Stone = len(os.listdir(os.path.join(train_dir, 'Stone')))
num_normal = len(os.listdir(os.path.join(train_dir, 'Normal')))
print(f"Stone={num_Stone}")
print(f"Normal={num_normal}")

print("\nTest set:")
print('-'*60)
print(f"Stone={len(os.listdir(os.path.join(test_dir, 'Stone')))}")
print(f"Normal={len(os.listdir(os.path.join(test_dir, 'Normal')))}")

print("\nValidation set")
print('-'*60)
print(f"Stone={len(os.listdir(os.path.join(val_dir, 'Stone')))}")
print(f"Normal={len(os.listdir(os.path.join(val_dir, 'Normal')))}")

Stone = os.listdir(train_dir + "Stone")
Stone_dir = train_dir + "Stone"



plt.figure(figsize=(15, 5))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(Stone_dir, Stone[i]))
    plt.title("Stone")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    

plt.tight_layout()







normal = os.listdir(train_dir + "Normal")
normal_dir = train_dir + "Normal"

plt.figure(figsize=(10, 5))

for i in range(9):
    plt.subplot(3, 3, i + 1)
    img = plt.imread(os.path.join(Stone_dir, Stone[i]))
    plt.title("Normal")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
plt.tight_layout()


import glob

Stone_train = glob.glob(train_dir+"/Stone/*.jpg")
normal_train = glob.glob(train_dir+"/Normal/*.jpg")


data = pd.DataFrame(np.concatenate([[0]*len(normal_train) , [1]*len(Stone_train)]),columns=["class"])


plt.figure(figsize=(15,10))
sns.countplot(data['class'],data=data,palette='rocket')
plt.title('Stone vs Normal')
plt.show()

img_Datagen = ImageDataGenerator(
        rescale = 1/255,
        shear_range=10,
        zoom_range=0.3,
        horizontal_flip=True,
        width_shift_range = 0.2,
        rotation_range=20,
        fill_mode = 'nearest'
)
val_Datagen = ImageDataGenerator(
        rescale = 1/255
)

train = img_Datagen.flow_from_directory(train_dir,
                                       batch_size=8,
                                       class_mode='binary',
#                                        target_size=(224,224,3))
                                       )

validation = val_Datagen.flow_from_directory(val_dir,
                                              batch_size=1,
                                              class_mode='binary',
#                                               target_size=(224,224,3))
                                            )

test = img_Datagen.flow_from_directory(test_dir,
                                       batch_size=1,
                                       class_mode='binary',
#                                        target_size=(224/,224,3))
                                      )

img, label = next(train)
img.shape

vgg_model = tf.keras.applications.VGG19(
    weights='imagenet',
    include_top = False,
#     input_shape = (224,224,3)
)

for layer in vgg_model.layers:
    layer.trainable=False
    
x = vgg_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
# output layer
predictions = tf.keras.layers.Dense(1,activation='sigmoid')(x)

model = tf.keras.Model(inputs=vgg_model.input, outputs=predictions)

# to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6)

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])


model.summary()

history = model.fit(train,epochs=1, 
                    validation_data=validation,
                     steps_per_epoch=100,
#                     callbacks=[early_stopping],
                    batch_size=32)

# Evaluating the model on traina and test
score = model.evaluate(train)

print("Train Loss: ", score[0])
print("Train Accuracy: ", score[1])


# Test data
score = model.evaluate(test)

print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1])

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')


mobileNet_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top = False,
    input_shape = (224,224,3)
)

for layer in mobileNet_model.layers:
    layer.trainable=False
    
x = mobileNet_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
# output layer
predictions = tf.keras.layers.Dense(1,activation='sigmoid')(x)

model2 = tf.keras.Model(inputs=mobileNet_model.input, outputs=predictions)

# to avoid overfitting
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=6)

# Compiling the model
model2.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

history = model2.fit(train,epochs=30, 
                    validation_data=validation,
                     steps_per_epoch=100,
#                     callbacks=[early_stopping],
                    batch_size=32)


# Evaluating the model on traina and test
score = model2.evaluate(train)

print("Train Loss: ", score[0])
print("Train Accuracy: ", score[1])

score = model2.evaluate(test)
print("\nTest loss: ", score[0])
print("Test Accuracy: ", score[1])


plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val_Loss')
plt.legend()
plt.title('Loss Evolution')

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')

