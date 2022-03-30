import os
from django.conf import settings
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent.parent
import os


def identify_images(image_path):
    from keras.models import model_from_json
    from keras.preprocessing import image
    from keras.applications.inception_v3 import preprocess_input
    from keras.applications.inception_v3 import decode_predictions
    from keras.applications.inception_v3 import InceptionV3
    import numpy as np
    import matplotlib.pyplot as plt

    # load json and create model
    json_file = open(base_dir + 'VGG16_pre_train_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(base_dir + 'VGG16_pre_train_weights.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    #img = image.load_img(image_path, target_size=(224, 224))
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = loaded_model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]

    # base_dir = os.path.join(BASE_DIR, 'static/data/')
    # model = base_dir + 'VGG16_pre_train_model.h5'
    # model = load_model(model)

    # img = image.load_img(image_path, target_size=(224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # preds = model.predict(x)
    # # decode the results into a list of tuples (class, description, probability)
    # # (one such list for each sample in the batch)
    # print('Predicted:', decode_predictions(preds, top=3)[0])
    # # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


def identify_images_with_h5_model(base_dir, image_path):
    from keras.models import model_from_json
    from keras.preprocessing import image
    from keras.applications.inception_v3 import preprocess_input
    from keras.applications.inception_v3 import decode_predictions
    from keras.applications.inception_v3 import InceptionV3
    import numpy as np
    import matplotlib.pyplot as plt

    # load json and create model
    json_file = open(base_dir + 'VGG16_pre_train_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(base_dir + 'VGG16_pre_train_weights.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['categorical_accuracy'])
    #img = image.load_img(image_path, target_size=(224, 224))
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = loaded_model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


base_dir = os.path.join(BASE_DIR, 'static/data/')
identify_images_with_h5_model(base_dir, base_dir + 'test_1.jpg')