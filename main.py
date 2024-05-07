import tensorflow as tf
from tensorflow import keras
import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background

print(keras.__version__)
print(tf.__version__)


#set_background('./bgs/digital_pathology.jpg')

# set title
st.title('Pathology classification')

# set header
st.text('Please upload a pathology image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/pathology_classifier.h5')

# load class names
with open('./model/pathology_labels.txt', 'r') as f:
    #class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    class_names = [a[1:] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    #st.image(image, use_column_width=True)
    col1, col2, col3, col4 = st.columns(4)

    with col2:
        st.image(image, width=400)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## Prediction: {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
