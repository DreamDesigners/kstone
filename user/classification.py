import os
from django.conf import settings


def train_model():
    labels = ["Normal", "Stone"]
    base_dir = os.path.join( settings.BASE_DIR, 'static/data/' )

    normal_images_dir = os.path.join( base_dir, labels[0] )
    stone_images_dir = os.path.join( base_dir, labels[1] )

    normal_images = os.listdir( normal_images_dir )
    stone_images = os.listdir( stone_images_dir )

    batch_size = 32
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

    datagen = ImageDataGenerator(
        rescale=1./255,
        samplewise_center=True,
        samplewise_std_normalization=True,
        validation_split=0.2) # set validation split

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        seed = 1,
        subset='training') # set as training data

    validation_generator = datagen.flow_from_directory(
        base_dir, # same directory as training data
        target_size=(224, 224),
        batch_size=batch_size,
        #early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto'),
        class_mode='categorical', #Determines the type of label arrays that are returned:"categorical" will be 2D one-hot encoded labels,
        seed = 1,
        color_mode='rgb',
        subset='validation') # set as validation data

    print(validation_generator.class_indices)
    print(validation_generator.classes)

    print("total normal images: ", len(normal_images))
    print("total stone images: ", len(stone_images))


    from keras.applications.inception_v3 import InceptionV3
    from keras.preprocessing import image
    from keras.models import Model
    from keras.layers import Dense, Dropout, Flatten
    from keras import backend as K
    from keras import applications
    from keras.optimizers import SGD

    # build the VGG16 network#layers + optimizer
    batch_size = 32
    import keras_metrics
    metrics= ['categorical_accuracy', keras_metrics.precision(), keras_metrics.recall()]

    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
    x = base_model.output
    x = Flatten(input_shape=base_model.output_shape[1:])(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(len(labels), activation = 'softmax') (x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:19]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=1e-4, momentum=0.9),metrics=metrics)

    print("model summary: ", model.summary())

    history1=model.fit_generator(    
        train_generator,
        steps_per_epoch = train_generator.samples // batch_size,
        validation_data = validation_generator, 
        validation_steps = validation_generator.samples // batch_size,
        class_weight = 'balanced',
        seed = 1;
        epochs = 10)

    from matplotlib import pyplot

    pyplot.plot(history1.history['categorical_accuracy'])
    pyplot.plot(history1.history['val_categorical_accuracy'])
    pyplot.title('Training and validation accuracy')
    pyplot.show()

    pyplot.plot(history1.history['loss'])
    pyplot.plot(history1.history['val_loss'])
    pyplot.title('Training and validation loss')
    pyplot.show()


train_model()