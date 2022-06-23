from PIL import Image
import os

import streamlit as st

if __name__ == '__main__':
    st.title('Welcome To Project Nautilus AI!')
    
    instructions = """
        Please upload an image. The image you upload will be fed
        through the Deep Neural Network in real-time
        and the output will be displayed to the screen.
        """
    
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    if file: 
        img = Image.open(file)
        st.title("Here is the image you've selected")
        st.image(img)
    else:
        pass
    
    