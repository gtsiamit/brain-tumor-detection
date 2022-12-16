import cv2
import numpy as np
from glob import glob
import os
from pathlib import Path
import pandas as pd
import argparse
from tensorflow.keras.models import load_model

FILEDIR = Path(__file__)
BASE_DIR = FILEDIR.parent
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128


def load_image_data(img_path_list):
    img_list = []
    filename_list = []

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (IMAGE_SIZE_X,IMAGE_SIZE_Y))
        img_list.append(img_resized)
        filename = img_path.split('/')[-1].split('.')[0]
        filename_list.append(filename)

    img_arr = np.array(img_list)
    filename_arr = np.array(filename_list)

    return img_arr, filename_arr


def load_dataset(data_path):
    jpg_path = data_path.joinpath('**/*.jpg')
    img_files = glob(jpg_path, recursive = True)

    X_arr, fname_arr = load_image_data(img_path_list=img_files)

    return X_arr, fname_arr


def load_model_from_path():
    model = load_model(filepath=MODEL_PATH)
    return model


def predict():
    
    X, fname = load_dataset(data_path=DATA_PATH)

    print('X.shape:', X.shape)

    print('Predict')
    model = load_model_from_path()
    y_pred_prob = model.predict(X)
    y_pred_int = y_pred_prob.round()

    y_pred_prob_df = pd.DataFrame( np.concatenate( [fname[..., np.newaxis], y_pred_prob], axis=1 ), columns=['fname', 'pred_prob'] )
    y_pred_int_df = pd.DataFrame( np.concatenate( [fname[..., np.newaxis], y_pred_int], axis=1 ), columns=['fname', 'pred'] )

    y_pred_prob_df.to_parquet(RESULTS_PATH.joinpath('y_pred_prob_all.parquet.gzip'), compression='gzip')
    y_pred_int_df.to_parquet(RESULTS_PATH.joinpath('y_pred_all.parquet.gzip'), compression='gzip')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=False, help='Path to data')
    parser.add_argument('--model_path', type=str, required=False, help='Path to saved model')
    args = parser.parse_args()

    global DATA_PATH, MODEL_PATH
    DATA_PATH = args.data_path
    if not DATA_PATH:
        DATA_PATH = BASE_DIR.joinpath('dataset/archive/pred')
    else:
        DATA_PATH = Path(DATA_PATH)
    
    MODEL_PATH = args.model_path
    if not MODEL_PATH:
        MODEL_PATH = FILEDIR.joinpath('results/train/model_full.h5')
    else:
        MODEL_PATH = Path(MODEL_PATH)
    global RESULTS_PATH
    RESULTS_PATH = FILEDIR.joinpath('results/predict')
    os.makedirs(RESULTS_PATH, exist_ok=True)

    predict()


if __name__=='__main__':
    main()
