import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import tensorflow as tf
import pandas as pd
import argparse
from model import build_model
from tensorflow.keras.models import save_model
import pickle

FILEDIR = Path(__file__).parent
BASE_DIR = FILEDIR.parent
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128
NUMBER_FOLDS = 10
EPOCHS = 10
BATCH_SIZE = 32


def load_image_data(img_path_list, label):
    img_list = []
    lbl_list = []
    filename_list = []

    for img_path in img_path_list:
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (IMAGE_SIZE_X,IMAGE_SIZE_Y))
        img_list.append(img_resized)
        filename = img_path.split('/')[-1].split('.')[0]
        filename_list.append(filename)
    
    lbl_list = [label] * len(img_list)

    img_arr = np.array(img_list)
    lbl_arr = np.array(lbl_list)
    filename_arr = np.array(filename_list)

    return img_arr, lbl_arr, filename_arr


def load_dataset(data_path):
    jpg_path = str(data_path.joinpath('**/*.jpg'))
    img_files = glob(jpg_path, recursive = True)
    yes_img_files = [file for file in img_files if 'yes' in file]
    no_img_files = [file for file in img_files if 'no' in file]

    X_yes, y_yes, fname_yes = load_image_data(img_path_list=yes_img_files, label='yes')
    X_no, y_no, fname_no = load_image_data(img_path_list=no_img_files, label='no')

    X_arr = np.concatenate([X_yes, X_no], axis=0)
    y_arr = np.concatenate([y_yes, y_no], axis=0)
    fname_arr = np.concatenate([fname_yes, fname_no], axis=0)

    return X_arr, y_arr, fname_arr


def plot_history(history_input, fold=None):

    model_history = pd.DataFrame.from_dict(history_input)
    model_history['epoch'] = model_history.index.to_numpy()+1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    fig.suptitle('Training accuracy and training loss', fontsize=12)
    
    ax1.plot(model_history.epoch.to_numpy(), model_history.accuracy.to_numpy(), color='blue', label='Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='best')
    
    ax2.plot(model_history.epoch.to_numpy(), model_history.loss.to_numpy(), color='green', label='Loss')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='best')
    
    plt.xlabel('Epoch')
    plt.xticks(ticks=model_history.epoch.to_list())

    if fold:
        fig.savefig(str(RESULTS_PATH.joinpath(f'history_fold_{fold}.png')), facecolor='white', transparent=False, bbox_inches='tight')
    else:
        fig.savefig(str(RESULTS_PATH.joinpath(f'history.png')), facecolor='white', transparent=False, bbox_inches='tight')


def plot_cm(cm_array, cm_title, xticks, yticks, fmt, filename):

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(cm_array, annot=True, cmap='Blues', fmt=fmt, annot_kws={'size': 12})
    ax.set_title(cm_title)
    ax.set_xlabel('\nPredicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.xaxis.set_ticklabels(xticks, fontsize=12)
    ax.yaxis.set_ticklabels(yticks, fontsize=12)
    fig.savefig(str(RESULTS_PATH.joinpath(f'{filename}.png')), facecolor='white', transparent=False, bbox_inches='tight')


def plot_confusion_matrix(y_true_all, y_pred_all, le):

    cm = confusion_matrix(y_true=le.inverse_transform(y_true_all), y_pred=le.inverse_transform(y_pred_all[:,0].astype(int)), labels=le.classes_)
    cm_norm = confusion_matrix(y_true=le.inverse_transform(y_true_all), y_pred=le.inverse_transform(y_pred_all[:,0].astype(int)), labels=le.classes_, normalize='true')
    cm_norm *=100

    acc = accuracy_score(y_true=y_true_all, y_pred=y_pred_all) * 100
    acc_title = f'Accuracy: {acc:.02f}%'

    plot_cm(cm_array=cm, cm_title=acc_title, xticks=le.classes_, yticks=le.classes_, fmt='g', filename='cm')
    plot_cm(cm_array=cm_norm, cm_title=acc_title, xticks=le.classes_, yticks=le.classes_, fmt='.02f', filename='cm_norm')


def save_model_locally(model, model_fname):
    save_model(model=model, filepath=str(RESULTS_PATH.joinpath(model_fname)))


def train():
    
    X, y, fname = load_dataset(data_path=DATASET_PATH)
    print('Dataset loaded')

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    with open(str(RESULTS_PATH.joinpath('le.pkl')), 'wb') as fid:
        pickle.dump(le, fid)

    skf = StratifiedKFold(n_splits=NUMBER_FOLDS, random_state=2, shuffle=True)
    
    y_true_all = []
    fname_folds_all = []
    y_pred_all = []
    y_pred_prob_all = []
    fold_num_all = []

    print('X.shape:', X.shape)
    model_input_shape = (X.shape[1],X.shape[2],X.shape[3])

    fold_num = 0
    for train_index, test_index in skf.split(X, y):
        fold_num +=1
        print(f'-- Fold {fold_num} --')

        X_train, X_test = X[train_index], X[test_index]
        y_enc_train, y_enc_test = y_enc[train_index], y_enc[test_index]
        fname_train, fname_test = fname[train_index], fname[test_index]

        model = build_model( input_shape=model_input_shape )
        history = model.fit(X_train, tf.cast(y_enc_train, tf.float32), verbose=1, epochs=EPOCHS, batch_size=BATCH_SIZE)
        plot_history(history_input=history.history, fold=fold_num)
        save_model_locally(model=model, model_fname=f'model_fold_{fold_num}.h5')
        
        y_pred_prob = model.predict(X_test)
        y_pred_prob_all.append(y_pred_prob)
        y_pred_int = y_pred_prob.round()
        y_pred_all.append(y_pred_int)
        y_true_all.append(y_enc_test)
        fname_folds_all.append(fname_test)
        fold_num_all.append( np.array( [fold_num] * len(X_test) ) )

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_pred_prob_all = np.concatenate(y_pred_prob_all, axis=0)
    fname_folds_all = np.concatenate(fname_folds_all, axis=0)
    fold_num_all = np.concatenate(fold_num_all, axis=0)

    plot_confusion_matrix(y_true_all=y_true_all, y_pred_all=y_pred_all, le=le)

    pd.DataFrame(y_pred_prob_all, columns=['pred_prob']).to_parquet(str(RESULTS_PATH.joinpath('y_pred_prob_all.parquet.gzip')), compression='gzip')
    pd.DataFrame(y_pred_all, columns=['pred']).to_parquet(str(RESULTS_PATH.joinpath('y_pred_all.parquet.gzip')), compression='gzip')
    pd.DataFrame(y_true_all, columns=['true']).to_parquet(str(RESULTS_PATH.joinpath('y_true_all.parquet.gzip')), compression='gzip')
    pd.DataFrame(fname_folds_all, columns=['fname']).to_parquet(str(RESULTS_PATH.joinpath('fname_folds_all.parquet.gzip')), compression='gzip')
    pd.DataFrame(fold_num_all, columns=['fold']).to_parquet(str(RESULTS_PATH.joinpath('fold_num_all.parquet.gzip')), compression='gzip')

    if SAVE_FULL_MODEL:
        print('-- Full train --')
        model = build_model( input_shape=model_input_shape )
        history = model.fit(X, tf.cast(y_enc, tf.float32), verbose=1, epochs=EPOCHS, batch_size=BATCH_SIZE)
        plot_history(history_input=history.history)
        save_model_locally(model=model, model_fname=f'model_full.h5')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=False, help='Path to dataset')
    parser.add_argument('--save_full_model', choices=('True','False'), type=str, required=False, default='True', help='Train model on full dataset')
    args = parser.parse_args()

    global DATASET_PATH, SAVE_FULL_MODEL
    SAVE_FULL_MODEL = args.save_full_model == 'True'
    DATASET_PATH = args.dataset_path
    if not DATASET_PATH:
        DATASET_PATH = BASE_DIR.joinpath('dataset')
    else:
        DATASET_PATH = Path(DATASET_PATH)
    
    global RESULTS_PATH
    RESULTS_PATH = FILEDIR.joinpath('results/train')
    os.makedirs(RESULTS_PATH, exist_ok=True)

    train()


if __name__=='__main__':
    main()
