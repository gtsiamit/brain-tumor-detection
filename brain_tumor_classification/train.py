import cv2
import numpy as np
from glob import glob
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import pandas as pd
import argparse
import sys

# Setting up paths
FILEDIR = Path(__file__).parent
ROOT_PATH = Path.cwd()

sys.path.append(os.path.join(str(ROOT_PATH), 'utils'))
from utils import save_pickle, plot_history, plot_confusion_matrix, save_model_locally

from model import build_model

# Setting up parameters
# Image shape
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128
# Training hyperparameters
NUMBER_FOLDS = 10
#EPOCHS = 10
EPOCHS = 1
BATCH_SIZE = 32


def load_image_data(img_path_list: list, label: str):
    """Loads image data for specific label

    Args:
        img_path_list (list): Image path list
        label (str): Label to load

    Returns:
        img_arr (np.array): Image data array
        lbl_arr (np.array): Label array
        filename_arr (np.array): Image filename array
    """

    img_list = []
    lbl_list = []
    filename_list = []

    for img_path in img_path_list:
        # load image
        img = cv2.imread(img_path)
        # resize image
        img_resized = cv2.resize(img, (IMAGE_SIZE_X,IMAGE_SIZE_Y))
        img_list.append(img_resized)
        # find and store image filename
        filename = img_path.split('/')[-1].split('.')[0]
        filename_list.append(filename)
    
    # create list with labels
    lbl_list = [label] * len(img_list)

    img_arr = np.array(img_list)
    lbl_arr = np.array(lbl_list)
    filename_arr = np.array(filename_list)

    return img_arr, lbl_arr, filename_arr


def load_dataset(data_path: Path):
    """Loads dataset

    Args:
        data_path (Path): Dataset path

    Returns:
        X_arr (np.array): Image data array
        y_arr (np.array): Label array
        fname_arr (np.array): Image filenames arrays
    """
    # find image files paths
    jpg_path = str(data_path.joinpath('**/*.jpg'))
    img_files = glob(jpg_path, recursive = True)
    # separate files according to label
    yes_img_files = [file for file in img_files if 'yes' in file]
    no_img_files = [file for file in img_files if 'no' in file]

    # load data according to label
    X_yes, y_yes, fname_yes = load_image_data(img_path_list=yes_img_files, label='yes')
    X_no, y_no, fname_no = load_image_data(img_path_list=no_img_files, label='no')

    # concatenate data
    X_arr = np.concatenate([X_yes, X_no], axis=0)
    y_arr = np.concatenate([y_yes, y_no], axis=0)
    fname_arr = np.concatenate([fname_yes, fname_no], axis=0)

    return X_arr, y_arr, fname_arr


def train():
    """Train flow for CNN classification
    """
    
    # load dataset, X, y, filenames
    X, y, fname = load_dataset(data_path=DATASET_PATH)
    print('Dataset loaded')

    # label encoding of y data
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # store label encoder
    le_store_path = str(RESULTS_PATH.joinpath('le.pickle'))
    save_pickle(fname=le_store_path, object=le)

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=NUMBER_FOLDS, random_state=2, shuffle=True)
    
    # create empty lists for appending results and data of each fold
    y_true_all = []
    fname_folds_all = []
    y_pred_all = []
    y_pred_prob_all = []
    fold_num_all = []

    # model input data shape
    print('X.shape:', X.shape)
    model_input_shape = (X.shape[1],X.shape[2],X.shape[3])

    fold_num = 0
    for train_index, test_index in skf.split(X, y):
        fold_num +=1
        print(f'-- Fold {fold_num} --')

        # get train and test data for each fold
        X_train, X_test = X[train_index], X[test_index]
        y_enc_train, y_enc_test = y_enc[train_index], y_enc[test_index]
        fname_train, fname_test = fname[train_index], fname[test_index]

        # build model
        model = build_model( input_shape=model_input_shape )
        # train model with training data and generate history plot
        history = model.fit(X_train, tf.cast(y_enc_train, tf.float32), verbose=1, epochs=EPOCHS, batch_size=BATCH_SIZE)
        plot_history(history=history, fullpath=RESULTS_PATH.joinpath(f'history_fold_{fold_num}.jpg'))
        # store model locally
        save_model_locally(model=model, model_path=RESULTS_PATH.joinpath(f'model_fold_{fold_num}.h5'))
        
        # make predictions on testing data
        y_pred_prob = model.predict(X_test)
        # append predictions to lists
        y_pred_prob_all.append(y_pred_prob)
        y_pred_int = y_pred_prob.round()
        y_pred_all.append(y_pred_int)
        # append true labels and other information to lists
        y_true_all.append(y_enc_test)
        fname_folds_all.append(fname_test)
        fold_num_all.append( np.array( [fold_num] * len(X_test) ) )

    # concatenate results of all folds
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_pred_prob_all = np.concatenate(y_pred_prob_all, axis=0)
    fname_folds_all = np.concatenate(fname_folds_all, axis=0)
    fold_num_all = np.concatenate(fold_num_all, axis=0)

    # generate and store confusion matrices
    plot_confusion_matrix(y_true=le.inverse_transform(y_true_all), y_pred=le.inverse_transform(y_pred_all[:,0].astype(int)), norm=False, fullpath=RESULTS_PATH.joinpath('cm.jpg'))
    plot_confusion_matrix(y_true=le.inverse_transform(y_true_all), y_pred=le.inverse_transform(y_pred_all[:,0].astype(int)), norm=True, fullpath=RESULTS_PATH.joinpath('normalized_cm.jpg'))

    # store results
    save_pickle(fname=str(RESULTS_PATH.joinpath('y_pred_prob_all.pickle')), object=pd.DataFrame(y_pred_prob_all, columns=['pred_prob']))
    save_pickle(fname=str(RESULTS_PATH.joinpath('y_pred_all.pickle')), object=pd.DataFrame(y_pred_all, columns=['pred']))
    save_pickle(fname=str(RESULTS_PATH.joinpath('y_true_all.pickle')), object=pd.DataFrame(y_true_all, columns=['true']))
    save_pickle(fname=str(RESULTS_PATH.joinpath('fname_folds_all.pickle')), object=pd.DataFrame(fname_folds_all, columns=['fname']))
    save_pickle(fname=str(RESULTS_PATH.joinpath('fold_num_all.pickle')), object=pd.DataFrame(fold_num_all, columns=['fold']))

    # check if full train is needed
    if SAVE_FULL_MODEL:
        print('-- Full train --')
        # build and train full model on full dataset
        model = build_model( input_shape=model_input_shape )
        history = model.fit(X, tf.cast(y_enc, tf.float32), verbose=1, epochs=EPOCHS, batch_size=BATCH_SIZE)
        # plot and store the history of full train
        plot_history(history=history, fullpath=RESULTS_PATH.joinpath(f'history.jpg'))
        # store full model locally
        save_model_locally(model=model, model_path=RESULTS_PATH.joinpath('model_full.h5'))


def main():

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_full_model', choices=('True','False'), type=str, required=False, default='True', help='Train model on full dataset')
    args = parser.parse_args()

    global DATASET_PATH, SAVE_FULL_MODEL
    # variable controls of a full model train is performed
    SAVE_FULL_MODEL = args.save_full_model == 'True'
    # dataset path
    DATASET_PATH = args.dataset_path
    DATASET_PATH = Path(DATASET_PATH)
    
    # set and create (if not exists) the output folder
    global RESULTS_PATH
    RESULTS_PATH = FILEDIR.joinpath('output/train')
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # run train flow
    train()


if __name__=='__main__':
    main()
