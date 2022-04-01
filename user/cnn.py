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
        dic = {
            0: "Normal", 
            1: "STONE"}
        plt.title(dic.get(Y_batch[0]))
        plt.axis("off")
        plt.imshow(np.squeeze(image),cmap="gray",interpolation="nearest")
        break
plt.tight_layout()
plt.show()




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

cnn = Sequential()
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(img_width, img_height, 1)))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Flatten())
cnn.add(Dense(activation = 'relu', units = 128))
cnn.add(Dense(activation = 'relu', units = 64))
cnn.add(Dense(activation = 'sigmoid', units = 1))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.summary()


#Hyperparameters of Conv2D
Conv2D(filters=32,
    kernel_size=9,
    strides=(1, 1),
    padding="valid",
    activation=None,
    )
# Hyperparameters of MaxPooling2D 
MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")

from tensorflow.keras.utils import plot_model
plot_model(cnn,show_shapes=True, show_layer_names=True, rankdir='TB', expand_nested=True)

early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
learning_rate_reduction = ReduceLROnPlateau(monitor="val_loss", patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
callbacks_list = [ early, learning_rate_reduction]

# from sklearn.utils.class_weight import compute_class_weight
# weights = compute_class_weight('balanced', np.unique(train.classes), train.classes)
# cw = dict(zip( np.unique(train.classes), weights))
# print(cw)

from sklearn.utils import compute_class_weight

class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train.classes),
                                        y = train.classes                                                    
                                    )
class_weights = dict(zip(np.unique(train.classes), class_weights))
print(class_weights)

cw = class_weights

cnn.fit(train,epochs=1, validation_data=valid, class_weight=cw, callbacks=callbacks_list)

fp =  base_dir + "model.h5"
cnn.save(fp)

from tensorflow.keras.models import load_model

cnn = load_model(fp)

cnn.predict_segmentation()
pd.DataFrame(cnn.history.history).plot()

test_accu = cnn.evaluate(test)
print('The testing accuracy is :',test_accu[1]*100, '%')

preds = cnn.predict(test,verbose=1)

predictions = preds.copy()
predictions[predictions <= 0.5] = 0
predictions[predictions > 0.5] = 1

from sklearn.metrics import classification_report,confusion_matrix

cm = pd.DataFrame(data=confusion_matrix(test.classes, predictions, labels=[0, 1]),
                  index=["Actual Normal", "Actual Stone"],
                  columns=["Predicted Normal", "Predicted Stone"])
import seaborn as sns
sns.heatmap(cm,annot=True,fmt="d")

print(classification_report(y_true=test.classes, y_pred=predictions,
                        target_names =['Normal','Stone']
))


test.reset()
x=np.concatenate([test.next()[0] for i in range(test.__len__())])
y=np.concatenate([test.next()[1] for i in range(test.__len__())])
print(x.shape)
print(y.shape)


dic = {0:'Normal', 1:'Stone'}
plt.figure(figsize=(20,20))
for i in range(0+228, 9+228):
  plt.subplot(3, 3, (i-228)+1)
  if preds[i, 0] >= 0.5: 
      out = ('{:.2%} probability of being Stone case'.format(preds[i][0]))
      
      
  else: 
      out = ('{:.2%} probability of being Normal case'.format(1-preds[i][0]))
      
      

  plt.title(out+"\n Actual case : "+ dic.get(y[i]))    
  plt.imshow(np.squeeze(x[i]))
  plt.axis('off')
plt.show()



# Testing with my own Chest X-Ray
hardik_path = base_dir + 'test_1.jpg'

from tensorflow.keras.preprocessing import image

hardik_img = image.load_img(hardik_path, target_size=(500, 500),color_mode='grayscale')

# Preprocessing the image
pp_hardik_img = image.img_to_array(hardik_img)
pp_hardik_img = pp_hardik_img/255
pp_hardik_img = np.expand_dims(pp_hardik_img, axis=0)

#predict
hardik_preds= cnn.predict(pp_hardik_img)

#print
plt.figure(figsize=(6,6))
plt.axis('off')
if hardik_preds>= 0.5: 
    out = ('I am {:.2%} percent confirmed that this is a Stone case'.format(hardik_preds[0][0]))
    
else: 
    out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(1-hardik_preds[0][0]))
    

plt.title("Hardik's Chest X-Ray\n"+out)  
plt.imshow(np.squeeze(pp_hardik_img))
plt.show()