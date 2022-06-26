from PIL import Image
import os

import streamlit as st

import tensorflow as tf

import numpy as np

def load_model(path_to_dir):
    reloaded = tf.keras.models.load_model(path_to_dir)
    return reloaded

def inference(image, model):
    img = tf.image.resize(image, [224,224])
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    return score

if __name__ == '__main__':
    st.title('Welcome To Project Nautilus AI!')
    
    instructions = """
        Please upload an image. The image you upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen. This is work in progress.
        """
    
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    if file: 
        img = Image.open(file)
        st.title("Here is the image you've selected")
        st.image(img)

        st.title("Inference Result")
        model = load_model('inference_graph\\1654458544')
        score = inference(img, model)
        class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
        st.write("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))
    else:
        pass
    
    
