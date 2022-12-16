import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from pathlib import Path
import pandas as pd
import argparse
from model import build_unet_model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

FILEDIR = Path(__file__)
BASE_DIR = FILEDIR.parent
IMAGE_SIZE_X = 128
IMAGE_SIZE_Y = 128
EPOCHS = 20
EPOCHS_ES_MC = 2*EPOCHS
BATCH_SIZE = 32
PATIENCE_ES = 5


def load_dataset(dataset_part):
    img_dir = DATASET_PATH.joinpath(f'archive/Br35H-Mask-RCNN/{dataset_part}')
    jpg_path = str(img_dir.joinpath('*.jpg'))
    img_files = glob(jpg_path, recursive = True)
    img_files = [f for f in img_files if 'mask' not in f]

    img_list = []
    fname_list = []
    mask_img_list = []
    mask_fname_list = []

    for img_path in img_files:
        fname = img_path.split('/')[-1].split('.')[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMAGE_SIZE_X,IMAGE_SIZE_Y))

        mask_fname = fname + '_mask'
        img_path_mask = img_path.replace(fname, mask_fname)
        img_mask = cv2.imread(img_path_mask, cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.resize(img_mask, (IMAGE_SIZE_X,IMAGE_SIZE_Y))

        img_list.append(img)
        fname_list.append(fname)
        mask_img_list.append(img_mask)
        mask_fname_list.append(mask_fname)
    
    img_arr = np.array(img_list)
    fname_arr = np.array(fname_list)
    img_arr_mask = np.array(mask_img_list)
    fname_arr_mask = np.array(mask_fname_list)

    img_arr = img_arr[..., np.newaxis]
    img_arr_mask = img_arr_mask[..., np.newaxis]

    return img_arr, fname_arr, img_arr_mask, fname_arr_mask


def normalize_image_data(input_array, inverse=False):
    if not inverse:
        normalized_array = cv2.normalize(src=input_array, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    else:
        normalized_array = cv2.normalize(src=input_array, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return normalized_array


def plot_history(history_input):

    model_history = pd.DataFrame.from_dict(history_input)
    model_history['epoch'] = model_history.index.to_numpy()+1
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle('Training and Validation metrics', fontsize=12)

    ax1.plot(model_history.epoch.to_numpy(), model_history.accuracy.to_numpy(), color='orange', label='Training Accuracy')
    ax1.plot(model_history.epoch.to_numpy(), model_history.val_accuracy.to_numpy(), color='blue', label='Validation Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend(loc='best')

    ax2.plot(model_history.epoch.to_numpy(), model_history.loss.to_numpy(), color='red', label='Training Loss')
    ax2.plot(model_history.epoch.to_numpy(), model_history.val_loss.to_numpy(), color='green', label='Validation Loss')
    ax2.set_ylabel('Loss')
    ax2.legend(loc='best')

    plt.xlabel('Epoch')
    plt.xticks(ticks=model_history.epoch.to_list())

    fig.savefig(str(RESULTS_PATH.joinpath('history.png')), facecolor='white', transparent=False, bbox_inches='tight')


def save_model_locally(model, model_fname):
    save_model(model=model, filepath=str(RESULTS_PATH.joinpath(model_fname)))


def load_stored_model(model_filepath):
    model_loaded = load_model(filepath=model_filepath)
    return model_loaded


def train():

    X_train, fname_X_train, y_train, fname_y_train = load_dataset(dataset_part='TRAIN')
    X_val, fname_X_val, y_val, fname_y_val = load_dataset(dataset_part='VAL')
    X_test, fname_X_test, y_test, fname_y_test = load_dataset(dataset_part='TEST')
    print('Dataset loaded')

    X_train, y_train = normalize_image_data(input_array=X_train), normalize_image_data(input_array=y_train)
    X_val, y_val = normalize_image_data(input_array=X_val), normalize_image_data(input_array=y_val)
    X_test, y_test = normalize_image_data(input_array=X_test), normalize_image_data(input_array=y_test)


    input_model_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_unet_model(input_shape=input_model_shape)


    if USE_CALLBACKS:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=PATIENCE_ES)
        store_model_path = str(RESULTS_PATH.joinpath('model_unet_es_mc.h5'))
        mc = ModelCheckpoint(store_model_path, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

        history = model.fit(x=X_train, y=y_train, validation_data=(X_val,y_val), epochs=EPOCHS_ES_MC, batch_size=BATCH_SIZE, callbacks=[es, mc], verbose=1)
    
    else:
        history = model.fit(x=X_train, y=y_train, validation_data=(X_val,y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    plot_history(history_input=history.history)

    if USE_CALLBACKS:
        model = load_stored_model(model_filepath=store_model_path)
    
    test_metrics = model.evaluate(X_test, y_test, verbose=1, batch_size=BATCH_SIZE)

    
    evaluation_results_filepath = str(RESULTS_PATH.joinpath('evaluation_metrics.txt'))
    with open(evaluation_results_filepath, 'w') as txt_file:
        print(f'Testing Loss: {test_metrics[0]}', file=txt_file)
        print(f'Testing Accuracy: {test_metrics[1]}', file=txt_file)


    if not USE_CALLBACKS:
        save_model_locally(model=model, model_fname=f'model_unet.h5')
    

    print('-- Finished --')


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, action='store', dest='dataset_path', required=False, help='Path to dataset')
    parser.add_argument('--use_callbacks', choices=('True','False'), type=str, action='store', dest='use_callbacks', required=False, default='True', help='Use callbacks for training model')
    args = parser.parse_args()

    global DATASET_PATH, USE_CALLBACKS
    USE_CALLBACKS = args.use_callbacks == 'True'
    DATASET_PATH = args.dataset_path
    if not DATASET_PATH:
        DATASET_PATH = BASE_DIR.joinpath('dataset/')
    else:
        DATASET_PATH = Path(DATASET_PATH)
    
    global RESULTS_PATH
    RESULTS_PATH = FILEDIR.joinpath('results/train/')
    os.makedirs(RESULTS_PATH, exist_ok=True)

    train()


if __name__=='__main__':
    main()

