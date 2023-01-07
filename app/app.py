import streamlit as st
from pathlib import Path
import argparse
import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import matplotlib.pyplot as plt

FILEDIR = Path(__file__).parent
BASE_DIR = FILEDIR.parent
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128


def app_input_form():

    with st.form(key="options_form"):

        file_uploaded = st.file_uploader(
            label="Choose an input file for the brain tumor detection system:", 
            type=['jpeg', 'jpg', 'png']
        )

        st.write("Choose the system parts to run")
        
        selectbox_classification = st.selectbox(
            label="Brain tumor classification:", 
            options=["Yes", "No"]
        )
        
        selectbox_segmentation = st.selectbox(
            label="Brain tumor segmentation:", 
            options=["Yes", "No"]
        )
        
        form_submitted = st.form_submit_button("Submit")

        if form_submitted:
            st.success('Successfully submitted')
        
        return file_uploaded, selectbox_classification, selectbox_segmentation


def store_uploaded_file(file_uploaded):

    img_path = UPLOAD_DIR.joinpath(file_uploaded.name)
    with open(str(img_path), "wb") as f:
        f.write(file_uploaded.getbuffer())


def load_image_file(filename):

    img_path = UPLOAD_DIR.joinpath(filename)

    img = cv2.imread( str(img_path) )
    img_resized = cv2.resize(img, (IMAGE_SIZE_X,IMAGE_SIZE_Y))

    return img_resized


def load_model_from_path(model_path):
    model = load_model(filepath=str(model_path))
    return model




def app_classification_results_output(img_arr):
    
    img_arr_for_pred = img_arr[np.newaxis,...]
    model = load_model_from_path(model_path=CLASSIFICATION_MODEL_PATH)

    pred_prob = model.predict(img_arr_for_pred, verbose=0)
    pred_int = pred_prob.round().astype(int)

    with open(str(LABEL_ENCODER_PATH), 'rb') as f:
        le = pickle.load(f)
    
    pred_prob_to_write = round(pred_prob[0][0]*100, 2) / 100
    pred_int_inverse = le.inverse_transform(pred_int[0])
    predicted_label = pred_int_inverse[0]

    st.header("Brain tumor classification results")

    st.write(f"Predicted probability for the input image is: {pred_prob_to_write}")
    st.write(f" The rounded predicted probability for the input image is: {pred_int[0][0]}")
    st.write(f" The predicted label for the input image is: {predicted_label}")

    st.image(image=img_arr, width=192, caption='Input image')

    if predicted_label=='yes':
        st.write(f"Brain tumor has been detected in the input image")
    elif predicted_label=='no':
        st.write(f"Brain tumor has not been detected in the input image")





def app():

    st.title("Welcome to Brain Tumor Detection system application")

    file_uploaded, selectbox_classification, selectbox_segmentation = app_input_form()

    if not file_uploaded is None:

        store_uploaded_file(file_uploaded=file_uploaded)

        img_filename = file_uploaded.name
        img_arr = load_image_file(filename=img_filename)

        if selectbox_classification=='Yes':
            app_classification_results_output(img_arr=img_arr)





def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_model_path', type=str, required=False, help='Path to classification model')
    parser.add_argument('--segmentation_model_path', type=str, required=False, help='Path to segmentation model')
    parser.add_argument('--label_encoder_path', type=str, required=False, help='Path to label encoder')
    args = parser.parse_args()

    global CLASSIFICATION_MODEL_PATH, SEGMENTATION_MODEL_PATH, LABEL_ENCODER_PATH
    CLASSIFICATION_MODEL_PATH = args.classification_model_path
    SEGMENTATION_MODEL_PATH = args.segmentation_model_path
    LABEL_ENCODER_PATH = args.label_encoder_path

    if not CLASSIFICATION_MODEL_PATH:
        CLASSIFICATION_MODEL_PATH = BASE_DIR.joinpath('brain_tumor_classification/results/train/model_full.h5')
    else:
        CLASSIFICATION_MODEL_PATH = Path(CLASSIFICATION_MODEL_PATH)
    if not SEGMENTATION_MODEL_PATH:
        SEGMENTATION_MODEL_PATH = BASE_DIR.joinpath('brain_tumor_segmentation/results/train/model_unet.h5')
    else:
        SEGMENTATION_MODEL_PATH = Path(SEGMENTATION_MODEL_PATH)
    if not LABEL_ENCODER_PATH:
        LABEL_ENCODER_PATH = BASE_DIR.joinpath('brain_tumor_classification/results/train/le.pkl')
    else:
        LABEL_ENCODER_PATH = Path(LABEL_ENCODER_PATH)
    
    global UPLOAD_DIR
    UPLOAD_DIR = FILEDIR.joinpath('upload_dir')
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    app()




if __name__=="__main__":
    main()



