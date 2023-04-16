from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Keras
import keras.utils as image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Flask
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

#Loading original model without class names in the output layer
model = load_model('VGG16model94accuracy.h5')

# Assigning output layer to a variable
output_layer=model.layers[-1]

# Set the class names attribute of the output layer's configuration to later save the model with the class names
output_layer_config = output_layer.get_config()
output_layer_config['class_names']=["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]
output_layer.config= output_layer_config

# Save the model with the class names attribute
model.save('VGG16model94accuracy_with_classes.h5')

# Loading the model with the class names attribute
model = load_model('VGG16model94accuracy_with_classes.h5')

# assigning the class names to indexes
class_indices = {0: 'meningioma_tumor', 1: 'glioma_tumor', 2: 'no_tumor', 3: 'pituitary_tumor'}
print('Model loaded.')
print('Running on http://localhost:5000')
# print(output_layer.config['class_names'])

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path


#Home route
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    #IF statement to check if the request is a POST request
    if request.method == 'POST':

        # Get the file path of the uploaded image and save it
        file_path = get_file_path_and_save(request)

        # Load the image using the Keras load_img function and resize it to (224 by 224)
        img = image.load_img(file_path, target_size=(224, 224))

        # Convert the image to a numpy array using the Keras img_to_array function
        img_data = image.img_to_array(img)

        # Add an extra dimension to the numpy array to create a batch of size 1 using np.expand_dims to allow the image to be processed by the model by itself and not in a batch
        img_data = np.expand_dims(img_data, axis=0)

        # Preprocess the image data using the preprocess_input function from Keras
        img_data = preprocess_input(img_data)

        # Use the model to predict the class probabilities for the input image and save it to a variable named preds
        preds = model.predict(img_data)

        #Print the class probabilities to see which class has the highest probability and what the lower probabilities are.
        #If it got it wrong, it would be nice to know what the other classes were and how close they were to the correct class.
        print(preds)

        # Get the index of the class with the highest probability using np.argmax
        class_idx = np.argmax(preds)

        # Get the name of the predicted class using the class_indices dictionary
        class_name = class_indices[class_idx]

        #Print the index of the class with the highest probability. If it doesn't match the index of the class name, then the class names are not in the correct order if we're using training data
        #as a benchmark for the order of the classes.
        # Using the iterative process is our best bet to ensure that the class names are in the correct order.
        print("Predicted class index:", class_idx)

        #Calculate the confidence level as a % by multiplying the probability by 100
        confidence = preds[0][class_idx] * 100

        #Print the predicted class name and the confidence level onto the screen
        result = f"Our model predicts that this image represents a {class_name} with a confidence level of {confidence:.2f}%"

        return result
    return None

if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()