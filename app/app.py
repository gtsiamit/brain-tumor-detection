import streamlit as st
from pathlib import Path
import argparse
import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import sys

# Setting up paths
FILEDIR = Path(__file__).parent
ROOT_PATH = Path.cwd()

sys.path.append(os.path.join(str(ROOT_PATH), "utils"))
from utils import normalize_image_data

# Setting up image shape
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128


def app_input_form():
    """Input form for app

    Returns:
        file_uploaded (st.runtime.uploaded_file_manager.UploadedFile): Uploaded file
        selectbox_classification (str): Controls if classification is performed
        selectbox_segmentation (str): Controls if segmentation is performed
    """

    with st.form(key="options_form"):

        # File uploader for input image
        file_uploaded = st.file_uploader(
            label="Choose an input file for the brain tumor detection system:", 
            type=['jpeg', 'jpg', 'png']
        )

        st.write("Choose the system parts to run")
        
        # Selectbox for classification task
        selectbox_classification = st.selectbox(
            label="Brain tumor classification:", 
            options=["Yes", "No"]
        )
        
        # Selectbox for segmentation task
        selectbox_segmentation = st.selectbox(
            label="Brain tumor segmentation:", 
            options=["Yes", "No"]
        )
        
        # Submit button
        form_submitted = st.form_submit_button("Submit")

        if form_submitted:
            st.success('Successfully submitted')
        
        return file_uploaded, selectbox_classification, selectbox_segmentation


def upload_testing_image(img_file_buffer: st.runtime.uploaded_file_manager.UploadedFile, grayscale=False):
    """Convert uploaded image to opencv image

    Args:
        img_file_buffer (st.runtime.uploaded_file_manager.UploadedFile): Uploaded image file
        grayscale (bool, optional): Load the image in grayscale or color. Defaults to False.

    Returns:
        img (Union[np.array, None]): The image array
    """
    if img_file_buffer is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        if not grayscale:
            img = cv2.imdecode(file_bytes, 1)
        else:
            img = cv2.imdecode(file_bytes, 0)
        # resize image
        img = cv2.resize(img, (IMAGE_SIZE_X,IMAGE_SIZE_Y))

        return img
    else:
        return None


@st.cache(allow_output_mutation=True)
def get_model(path: Path):
    """Gets the path, loads and returns the model

    Args:
        path (Path): The path of the model

    Returns:
        tf.keras.Model: The loaded model
    """

    return load_model(path)


def app_classification_results_output(img: np.array) -> str:
    """Extracts the classification results

    Args:
        img (np.array): Image array

    Returns:
        predicted_label (str): Predicted label from classification model
    """

    # Image preprocessing before classification
    processed_img = img[np.newaxis,...]

    # Load prestrained classification model
    model = get_model(CLASSIFICATION_MODEL_PATH)

    # Make prediction
    prediction = model.predict(processed_img, verbose=0)
    prob=prediction[0][0]

    if prob>=0.5:
        predicted_label='yes'
    else:
        predicted_label='no'
    
    pred_int = prob.round().astype(int)
    pred_prob_to_write = round(prob*100, 2) / 100
    pred_prob_to_write = f"{pred_prob_to_write:.4f}"

    # Print classification results
    st.header("Brain tumor classification results")

    st.write(f"Predicted probability for the input image is: {pred_prob_to_write}")
    st.write(f" The rounded predicted probability for the input image is: {pred_int}")
    st.write(f" The predicted label for the input image is: {predicted_label}")

    st.image(image=img, width=192, caption='Input image')

    # Generate the predicted label
    if predicted_label=='yes':
        st.write(f"Brain tumor has been detected in the input image")
    elif predicted_label=='no':
        st.write(f"Brain tumor has not been detected in the input image")
    
    return predicted_label


def app_segmentation_results_output(img: np.array):
    """Extracts the segmentation results

    Args:
        img (np.array): Image data array
    """
    
    # Image preprocessing before classification
    two_channel_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = two_channel_img[np.newaxis, ..., np.newaxis]
    processed_img_norm = normalize_image_data(input_array=processed_img)

    # Load prestrained classification model
    model = get_model(SEGMENTATION_MODEL_PATH)

    # Make prediction
    predicted_mask = model.predict(processed_img_norm, verbose=0)
    predicted_mask_norm_inv = normalize_image_data(input_array=predicted_mask, inverse=True)

    # Convert the image to 3-channel image
    pred_img_arr_3_channel = cv2.cvtColor(predicted_mask_norm_inv[0], cv2.COLOR_GRAY2RGB)

    # Apply the predicted mask to the image
    ret, thresh = cv2.threshold(src=pred_img_arr_3_channel, thresh=150, maxval=255, type=cv2.THRESH_BINARY)
    indices_above_thres = np.where(thresh>0)

    segmented_img = cv2.cvtColor(processed_img[0], cv2.COLOR_GRAY2RGB)
    segmented_img[indices_above_thres[:2]] = [255,0,0]

    # Print segmentation results
    st.header("Brain tumor segmentation results")
    st.write(f"Segmentation results")
    st.image([two_channel_img, segmented_img], caption=['Input image', 'Segmented image'], channels='RGB')


def app():
    """Executes the app for uploading images and generating classification and/or segmentantion results
    """

    st.title("Welcome to Brain Tumor Detection system application")

    file_uploaded, selectbox_classification, selectbox_segmentation = app_input_form()

    if not file_uploaded is None:

        cv_img = upload_testing_image(img_file_buffer=file_uploaded)

        # check if classifiction is requested
        if selectbox_classification=='Yes':

            #img_arr = load_image_file(filename=img_filename)
            predicted_classification_label = app_classification_results_output(img=cv_img)
        
            # check if segmentation is requested
            # if the classification result is 'yes', segmentation is performed
            if selectbox_segmentation=='Yes' and predicted_classification_label=='yes':
                app_segmentation_results_output(img=cv_img)
            
            # if the classification result is 'no', segmentation is not performed
            elif selectbox_segmentation=='Yes' and predicted_classification_label=='no':
                st.write(f"Brain tumor segmentation will not be applied on the input image.")


def main():

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_model_path', type=str, required=False, help='Path to classification model')
    parser.add_argument('--segmentation_model_path', type=str, required=False, help='Path to segmentation model')
    args = parser.parse_args()

    # paths to classification model and segmentation model
    global CLASSIFICATION_MODEL_PATH, SEGMENTATION_MODEL_PATH
    CLASSIFICATION_MODEL_PATH = args.classification_model_path
    SEGMENTATION_MODEL_PATH = args.segmentation_model_path
    
    if not CLASSIFICATION_MODEL_PATH:
        CLASSIFICATION_PATH = ROOT_PATH.joinpath('brain_tumor_classification/output/train')
        CLASSIFICATION_MODEL_PATH = CLASSIFICATION_PATH.joinpath('model_full.h5')
    if not SEGMENTATION_MODEL_PATH:
        SEGMENTATION_PATH = ROOT_PATH.joinpath('brain_tumor_segmentation/output/train')
        SEGMENTATION_MODEL_PATH = SEGMENTATION_PATH.joinpath('model_unet.h5')

    CLASSIFICATION_MODEL_PATH = Path(CLASSIFICATION_MODEL_PATH)
    SEGMENTATION_MODEL_PATH = Path(SEGMENTATION_MODEL_PATH)


    app()




if __name__=="__main__":
    main()
