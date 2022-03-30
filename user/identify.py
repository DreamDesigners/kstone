import os
from django.conf import settings
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent.parent
import os


def identify_image():
    `from keras.models import model_from_json
    from keras.preprocessing import image
    from keras.applications.inception_v3 import preprocess_input
    from keras.applications.inception_v3 import InceptionV3
    import numpy as np
    import matplotlib.pyplot as plt

    base_dir = os.path.join(BASE_DIR, 'static/data/')
    model = base_dir + 'VGG16_pre_train_model.json'
    weights = base_dir + 'VGG16_pre_train_weights.h5'
    test_image = base_dir + 'test_2.jpg'

    json_file = open(model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights)
    print("Loaded model from disk")

    # load an image to test the model
    img = image.load_img(test_image, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    preds = loaded_model.predict(x)

    print(preds)`

    # import random
    # from tensorflow.keras.preprocessing.image import img_to_array, load_img
    # # Let's define a new Model that will take an image as input, and will output
    # # intermediate representations for all layers in the previous model after
    # # the first.
    # successive_outputs = [layer.output for layer in model.layers[1:]]
    # visualization_model = Model(base_model.input, successive_outputs)

    # # Let's prepare a random input image of a cat or dog from the training set.
    # cat_img_files = [os.path.join(ate_dir, f) for f in ate_fnames]
    # dog_img_files = [os.path.join(car_dir, f) for f in car_fnames]
    # img_path = random.choice(cat_img_files + dog_img_files)

    # img = load_img(img_path, target_size=target_size)  # this is a PIL image
    # x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
    # x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

    # # Rescale by 1/255
    # x /= 255

    # # Let's run our image through our network, thus obtaining all
    # # intermediate representations for this image.
    # successive_feature_maps = visualization_model.predict(x)

    # # These are the names of the layers, so can have them as part of our plot
    # layer_names = [layer.name for layer in model.layers]

    # # Now let's display our representations
    # for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    #     if len(feature_map.shape) == 4:
    #         # Just do this for the conv / maxpool layers, not the fully-connected layers
    #         n_features = feature_map.shape[-1]  # number of features in feature map
    #         # The feature map has shape (1, size, size, n_features)
    #         size = feature_map.shape[1]
    #         # We will tile our images in this matrix
    #         display_grid = np.zeros((size, size * n_features))
    #         for i in range(n_features):
    #             # Postprocess the feature to make it visually palatable
    #             x = feature_map[0, :, :, i]
    #             x -= x.mean()
    #             x /= x.std()
    #             x *= 64
    #             x += 128
    #             x = np.clip(x, 0, 255).astype('uint8')
    #             # We'll tile each filter into this big horizontal grid
    #             display_grid[:, i * size : (i + 1) * size] = x
    #             # Display the grid
    #             scale = 20. / n_features
    #             pyplot.figure(figsize=(scale * n_features, scale))
    #             pyplot.title(layer_name)
    #             pyplot.grid(False)
    #             pyplot.imshow(display_grid, aspect='auto', cmap='viridis')

    



identify_image()