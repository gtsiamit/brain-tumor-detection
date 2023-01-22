import streamlit as st
from pathlib import Path
import argparse
import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import matplotlib.pyplot as plt
from PIL import Image

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


def remove_uploaded_file(file_uploaded):

    img_path = UPLOAD_DIR.joinpath(file_uploaded.name)
    os.remove(path=img_path)


def load_image_file(filename, grayscale=False):

    img_path = UPLOAD_DIR.joinpath(filename)

    if not grayscale:
        img = cv2.imread( str(img_path) )
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
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
    
    return predicted_label


def normalize_image_data(input_array, inverse=False):
    if not inverse:
        normalized_array = cv2.normalize(src=input_array, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        normalized_array = cv2.normalize(src=input_array, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return normalized_array


def plot_store_segmentation_resuts(input_img, seg_img, temp_img_path):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
    ax[0].imshow(X=input_img, cmap='gray')
    ax[0].set_title(label='Input image', fontsize=10)
    ax[0].axis('off')
    ax[1].imshow(X=seg_img)
    ax[1].set_title(label='Segmented image', fontsize=10)
    ax[1].axis('off')
    
    fig.savefig(temp_img_path, facecolor='white', transparent=False, bbox_inches='tight')


def app_segmentation_results_output(img_arr):

    model = load_model_from_path(model_path=SEGMENTATION_MODEL_PATH)

    img_arr_for_pred = img_arr[np.newaxis, ..., np.newaxis]
    img_arr_for_pred_norm = normalize_image_data(input_array=img_arr_for_pred)

    pred_img_arr = model.predict(img_arr_for_pred_norm, verbose=0)
    pred_img_arr_norm_inv = normalize_image_data(input_array=pred_img_arr, inverse=True)

    pred_img_arr_3_channel = cv2.cvtColor(pred_img_arr_norm_inv[0], cv2.COLOR_GRAY2RGB)

    ret, thresh = cv2.threshold(src=pred_img_arr_3_channel, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
    indices_above_thres = np.where(thresh>0)

    segmented_img = cv2.cvtColor(img_arr_for_pred[0], cv2.COLOR_GRAY2RGB)
    segmented_img[indices_above_thres[:2]] = [255,0,0]

    st.header("Brain tumor segmentation results")
    st.write(f"Input image and segmented image:")

    temp_img_path = str(TEMP_DIR.joinpath('temp_segmented_image.png'))
    plot_store_segmentation_resuts(input_img=img_arr_for_pred[0], seg_img=segmented_img, temp_img_path=temp_img_path)
    image_for_plot = Image.open(fp=temp_img_path)
    st.image(image=image_for_plot, caption='Segmentation results', channels='RGB')
    os.remove(path=temp_img_path)




def app():

    st.title("Welcome to Brain Tumor Detection system application")

    file_uploaded, selectbox_classification, selectbox_segmentation = app_input_form()

    if not file_uploaded is None:

        store_uploaded_file(file_uploaded=file_uploaded)

        img_filename = file_uploaded.name

        if selectbox_classification=='Yes':
            img_arr = load_image_file(filename=img_filename)
            predicted_classification_label = app_classification_results_output(img_arr=img_arr)
        
        if selectbox_segmentation=='Yes' and predicted_classification_label=='yes':
            img_arr = load_image_file(filename=img_filename, grayscale=True)
            app_segmentation_results_output(img_arr=img_arr)
        elif selectbox_segmentation=='Yes' and predicted_classification_label=='no':
            st.write(f"Brain tumor segmentation will not be applied on the input image.")
        
        remove_uploaded_file(file_uploaded=file_uploaded)




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
    
    global UPLOAD_DIR, TEMP_DIR
    UPLOAD_DIR = FILEDIR.joinpath('upload_dir')
    TEMP_DIR = FILEDIR.joinpath('temp_dir')
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    app()




if __name__=="__main__":
    main()
