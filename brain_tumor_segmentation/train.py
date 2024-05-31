import cv2
import numpy as np
from glob import glob
import os
from pathlib import Path
import argparse
from tensorflow.keras.callbacks import EarlyStopping
import sys

# Setting up paths
FILEDIR = Path(__file__).parent
ROOT_PATH = Path.cwd()

sys.path.append(os.path.join(str(ROOT_PATH), 'utils'))
from utils import plot_history, save_model_locally, normalize_image_data

from model import build_unet_model

# Setting up image shape
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128
# Setting up training hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE_ES = 3


def load_dataset(dataset_part: str):
    """Loads dataset

    Args:
        dataset_part (str): Dataset part to load e.g. TRAIN, VAL, TEST

    Returns:
        img_arr (np.array): Image data array
        fname_arr (np.array): Image filenames array
        img_arr_mask (np.array): Image masks data array
        fname_arr_mask (np.array): Image masks filenames array
    """
    # find paths of image files
    img_dir = DATASET_PATH.joinpath(dataset_part)
    jpg_path = str(img_dir.joinpath('*.jpg'))
    img_files = glob(jpg_path, recursive = True)
    img_files = [f for f in img_files if 'mask' not in f]

    img_list = []
    fname_list = []
    mask_img_list = []
    mask_fname_list = []

    for img_path in img_files:
        # find image filename
        fname = img_path.split('/')[-1].split('.')[0]
        # load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # resize image
        img = cv2.resize(img, (IMAGE_SIZE_X,IMAGE_SIZE_Y))

        # find mask image filepath
        mask_fname = fname + '_mask'
        img_path_mask = img_path.replace(fname, mask_fname)
        # load mask image
        img_mask = cv2.imread(img_path_mask, cv2.IMREAD_GRAYSCALE)
        # resize mask image
        img_mask = cv2.resize(img_mask, (IMAGE_SIZE_X,IMAGE_SIZE_Y))

        img_list.append(img)
        fname_list.append(fname)
        mask_img_list.append(img_mask)
        mask_fname_list.append(mask_fname)
    
    # store data to arrays
    img_arr = np.array(img_list)
    fname_arr = np.array(fname_list)
    img_arr_mask = np.array(mask_img_list)
    fname_arr_mask = np.array(mask_fname_list)

    img_arr = img_arr[..., np.newaxis]
    img_arr_mask = img_arr_mask[..., np.newaxis]

    return img_arr, fname_arr, img_arr_mask, fname_arr_mask


def train():
    """Train flow for UNET segmentation
    """

    # X is the initial image set and y is the mask image set
    # load training data
    X_train, fname_X_train, y_train, fname_y_train = load_dataset(dataset_part='TRAIN')
    # load validation data
    X_val, fname_X_val, y_val, fname_y_val = load_dataset(dataset_part='VAL')
    # load testing data
    X_test, fname_X_test, y_test, fname_y_test = load_dataset(dataset_part='TEST')
    print('Dataset loaded')

    # normalize image data
    X_train, y_train = normalize_image_data(input_array=X_train), normalize_image_data(input_array=y_train)
    X_val, y_val = normalize_image_data(input_array=X_val), normalize_image_data(input_array=y_val)
    X_test, y_test = normalize_image_data(input_array=X_test), normalize_image_data(input_array=y_test)


    # input shape of model
    input_model_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    # build model
    model = build_unet_model(input_shape=input_model_shape)


    # set early-stopping and train model with training data
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE_ES)
    history = model.fit(x=X_train, y=y_train, validation_data=(X_val,y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=1)

    # generate and store training history plot
    plot_history(history=history, fullpath=RESULTS_PATH.joinpath('history.jpg'))
    
    # evaluate model on testing data
    test_metrics = model.evaluate(X_test, y_test, verbose=1, batch_size=BATCH_SIZE)

    
    # store evaluation results
    evaluation_results_filepath = str(RESULTS_PATH.joinpath('evaluation_metrics.txt'))
    with open(evaluation_results_filepath, 'w') as txt_file:
        print(f'Testing Loss: {test_metrics[0]}', file=txt_file)
        print(f'Testing Accuracy: {test_metrics[1]}', file=txt_file)


    # save trained model locally
    save_model_locally(model=model, model_path=RESULTS_PATH.joinpath('model_unet.h5'))
    

    print('-- Finished --')


def main():

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, action='store', dest='dataset_path', required=False, help='Path to dataset')
    args = parser.parse_args()

    # path to dataset
    global DATASET_PATH
    DATASET_PATH = args.dataset_path
    if not DATASET_PATH:
        DATASET_PATH = ROOT_PATH.joinpath('dataset/Br35H-Mask-RCNN/')
    else:
        DATASET_PATH = Path(DATASET_PATH)
    
    # set and create (if not exists) the results folder
    global RESULTS_PATH
    RESULTS_PATH = FILEDIR.joinpath('output/train')
    os.makedirs(RESULTS_PATH, exist_ok=True)

    train()


if __name__=='__main__':
    main()

