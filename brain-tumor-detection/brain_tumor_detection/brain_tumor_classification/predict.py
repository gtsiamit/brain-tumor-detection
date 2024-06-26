import cv2
import numpy as np
from glob import glob
import os
from pathlib import Path
import pandas as pd
import argparse
import sys

# Setting up paths
FILEDIR = Path(__file__).parent
ROOT_PATH = Path.cwd()

sys.path.append(os.path.join(str(ROOT_PATH), "utils"))
from utils import save_pickle, load_model_from_path

# Setting up image shape
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128


def load_image_data(img_path_list: list):
    """Loads image data from specific paths

    Args:
        img_path_list (list): Image path list

    Returns:
        img_arr (np.array): Image data array
        filename_arr (np.array): Image filename array
    """
    img_list = []
    filename_list = []

    for img_path in img_path_list:
        # load image
        img = cv2.imread(img_path)
        # resize image
        img_resized = cv2.resize(img, (IMAGE_SIZE_X, IMAGE_SIZE_Y))
        img_list.append(img_resized)
        # find and store image filename
        filename = img_path.split("/")[-1].split(".")[0]
        filename_list.append(filename)

    img_arr = np.array(img_list)
    filename_arr = np.array(filename_list)

    return img_arr, filename_arr


def load_dataset(data_path: Path):
    """Loads dataset for predict

    Args:
        data_path (Path): Data path

    Returns:
        X_arr (np.array): Image data array
        fname_arr (np.array): Image filename array
    """
    # find image files paths
    jpg_path = data_path.joinpath("**/*.jpg")
    img_files = glob(str(jpg_path), recursive=True)

    # load image data
    X_arr, fname_arr = load_image_data(img_path_list=img_files)

    return X_arr, fname_arr


def predict():
    """Predict process with CNN classification model"""

    # load dataset
    X, fname = load_dataset(data_path=DATA_PATH)

    print("X.shape:", X.shape)

    print("Predict")
    # load pre-trained model
    model = load_model_from_path(model_path=MODEL_PATH)
    # make predictions
    y_pred_prob = model.predict(X)
    y_pred_int = y_pred_prob.round()

    y_pred_prob_df = pd.DataFrame(
        np.concatenate([fname[..., np.newaxis], y_pred_prob], axis=1),
        columns=["fname", "pred_prob"],
    )
    y_pred_int_df = pd.DataFrame(
        np.concatenate([fname[..., np.newaxis], y_pred_int], axis=1),
        columns=["fname", "pred"],
    )

    # store predictions
    save_pickle(
        fname=RESULTS_PATH.joinpath("y_pred_prob_all.pickle"), object=y_pred_prob_df
    )
    save_pickle(fname=RESULTS_PATH.joinpath("y_pred_all.pickle"), object=y_pred_int_df)


def main():

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument(
        "--model_path", type=str, required=False, help="Path to saved model"
    )
    args = parser.parse_args()

    global DATA_PATH, MODEL_PATH
    # dataset path
    DATA_PATH = args.data_path
    DATA_PATH = Path(DATA_PATH)

    # path to pre-trained model
    MODEL_PATH = args.model_path
    if not MODEL_PATH:
        MODEL_PATH = FILEDIR.joinpath("output/train/model_full.h5")
    else:
        MODEL_PATH = Path(MODEL_PATH)

    # set and create (if not exists) the output folder
    global RESULTS_PATH
    RESULTS_PATH = FILEDIR.joinpath("output/predict")
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # run predict flow
    predict()


if __name__ == "__main__":
    main()
