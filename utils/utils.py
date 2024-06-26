import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Union
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import save_model, load_model
import tensorflow as tf
import cv2


def load_json(path: Union[str, Path]) -> dict:
    """Loads a json file

    Parameters
    ----------
    path : Union[str, Path]
        Json file fullpath

    Returns
    -------
    dict
        Loaded json file
    """
    with open(path, "r") as f:
        annot_dict = json.load(f)
    return annot_dict


def save_pickle(fname: Union[str, Path], object):
    """Stores an object in pickle format

    Parameters
    ----------
    fname : Union[str, Path]
        export full path (with filename)
    object
        It can be any object
    """

    if isinstance(fname, str):
        fname = Path(fname)

    fname.parent.mkdir(parents=True, exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(object, f)


def load_pickle(fname: Union[str, Path]):
    """Load pickle file

    Parameters
    ----------
    fname : Union[str, Path]
        Full file path

    Returns
    -------
        The loaded object
    """

    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    norm: bool,
    fullpath: Union[str, Path],
    figsize=(8, 5),
):
    """Plot and store as image the confusion matrix

    Parameters
    ----------
    y_true : np.ndarray
        Groundtruth
    y_pred : np.ndarray
        Prediction
    norm : bool
        Normalize or not the confusion matrix
    fullpath : Union[str, Path]
        Export full path
    figsize : tuple, optional
        Image size, by default (8,5)
    """

    if not isinstance(fullpath, Path):
        fullpath = Path(fullpath)

    labels = np.unique(y_true).tolist()
    if norm:
        cm = confusion_matrix(y_true, y_pred, normalize="true") * 100
        fmt = "1.2f"
    else:
        cm = confusion_matrix(y_true, y_pred)
        fmt = "d"

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    acc = round(accuracy_score(y_true, y_pred) * 100, 2)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_df,
        annot=True,
        cmap="Blues",
        cbar=False,
        fmt=fmt,
        annot_kws={"size": 35 / np.sqrt(len(cm_df))},
    )
    plt.ylabel("Actal Values")
    plt.yticks(rotation=0)
    plt.xlabel("Predicted Values")
    plt.title(f"Accuracy: {acc}%", fontsize=15)
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fullpath), transparent=True, bbox_inches="tight")
    plt.show(block=False)


def plot_history(history, fullpath):
    """Plots the training history and stores it as image file"""

    if not isinstance(fullpath, Path):
        fullpath = Path(fullpath)

    epochs = len(history.history["accuracy"])
    xticks = np.arange(1, epochs + 1, 1)

    fig, axs = plt.subplots(2)
    # Create accuracy subplot
    axs[0].plot(xticks, history.history["accuracy"], label="Train accuracy")
    if "val_accuracy" in history.history.keys():
        axs[0].plot(xticks, history.history["val_accuracy"], label="Val accuracy")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # Create error subplot
    axs[1].plot(xticks, history.history["loss"], label="Train error")
    if "val_loss" in history.history.keys():
        axs[1].plot(xticks, history.history["val_loss"], label="Val error")
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.tight_layout()
    fullpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(fullpath), transparent=False, bbox_inches="tight")
    plt.show(block=False)


def save_model_locally(model: tf.keras.Model, model_path: Union[str, Path]):
    """Saves a trained model to a specific path

    Args:
        model (tf.keras.Model):
            The trained model
        model_path (Union[str, Path]):
            Path to save the model
    """

    if not isinstance(model_path, str):
        model_path = str(model_path)

    save_model(model=model, filepath=model_path)


def load_model_from_path(model_path: Union[str, Path]):
    """Loads a saved model from a specific path

    Args:
        model_path (Union[str, Path]):
            Path that the model is stored

    Returns:
        model (tf.keras.Model): The loaded model
    """

    if not isinstance(model_path, str):
        model_path = str(model_path)

    model = load_model(filepath=model_path)
    return model


def normalize_image_data(input_array: np.array, inverse: bool = False) -> np.array:
    """Normalizes (and inverse) the image data i.e. (0,1) or (0,255)

    Args:
        input_array (np.array): Image data array
        inverse (bool, optional): Controls the type of normalization. Defaults to False.

    Returns:
        normalized_array (np.array): Normalized image data array
    """
    if not inverse:
        normalized_array = cv2.normalize(
            src=input_array,
            dst=None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
    else:
        normalized_array = cv2.normalize(
            src=input_array,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )

    return normalized_array
