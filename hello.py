from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained VGG16 model
model_vgg16 = VGG16(weights='imagenet')

# Load the pre-trained VGG19 model
model_vgg19 = VGG19(weights='imagenet')

# Load and preprocess an image
img_path = "dataset/Patient Data/p1/fusion.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Predict the class of the image using VGG16
predictions_vgg16 = model_vgg16.predict(img_array)
decoded_predictions_vgg16 = decode_predictions(predictions_vgg16, top=3)[0]
print("VGG16 Predictions:", decoded_predictions_vgg16)

# Predict the class of the image using VGG19
predictions_vgg19 = model_vgg19.predict(img_array)
decoded_predictions_vgg19 = decode_predictions(predictions_vgg19, top=3)[0]
print("VGG19 Predictions:", decoded_predictions_vgg19)
