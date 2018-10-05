"""Useful functions for I/O actions."""

import os
import rasterio
import sys
import errno
import pickle
import keras.models

from .config import TILES_DIR, WATER_BITMAPS_DIR, WGS84_DIR, LABELS_DIR, MODELS_DIR, OUTPUT_DIR, TENSORBOARD_DIR

def create_directories():
    """Create all the directories in the /data directories which are used for preprocessing/training/evaluating."""

    directories = [TILES_DIR, WATER_BITMAPS_DIR, WGS84_DIR, LABELS_DIR, MODELS_DIR, OUTPUT_DIR, TENSORBOARD_DIR]
    for directory in directories:
        save_makedirs(directory)

def save_makedirs(path):
    """Create a directory and don't throw an exception if the
    directory doesn't exist yet.
    See http://stackoverflow.com/questions/273192/how-to-check-if-a-directory-exists-and-create-it-if-necessary"""

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def get_file_name(file_path):
    """Extract the file name without the file extension from a file path."""

    basename = os.path.basename(file_path)
    # Make sure we don't include the file extension.
    return os.path.splitext(basename)[0]


def save_model_summary(hyperparameters, model, path):
    """Save the hyperparameters of a model and the model summary generated by keras to a .txt file."""

    with open(os.path.join(path, "hyperparameters.txt"), "wb") as out:
        for parameter, value in hyperparameters:
            out.write("{}: {}\n".format(parameter, value).encode())

        # model.summary() prints to stdout. Because we want to write the
        # summary to a file we have to set the stdout to the file.
        stdout = sys.stdout
        sys.stdout = out
        model.summary
        sys.stdout = stdout


def save_tiles(file_path, tiled_features, tiled_labels):
    """Save the tile data for a satellite image as a pickle."""

    print("Store tile data at {}.".format(file_path))
    with open(file_path, "wb") as out:
        pickle.dump({"features": tiled_features, "labels": tiled_labels}, out)


def load_bitmap(file_path):
    """Load a GeoTIFF which is a bitmap of our water and non-water features."""
    with rasterio.open(file_path) as src:
        bitmap = src.read(1)

    return bitmap

def save_bitmap(file_path, image, source):
    """Save a bitmap given as a 2D matrix as a GeoTIFF."""

    print("Save result at {}.".format(file_path))
    with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            dtype=rasterio.uint8,
            count=1,
            width=source.width,
            height=source.height,
            transform=source.transform) as dst:
        dst.write(image, indexes=1)


def load_model(model_id):
    """Load a keras model and its weights with the given ID."""

    model_dir = os.path.join(MODELS_DIR, model_id)

    print("Load model in {}.".format(model_dir))
    model_file = os.path.join(model_dir, "model.json")
    with open(model_file, "r") as f:
        json_file = f.read()
        model = keras.models.model_from_json(json_file)

    weights_file = os.path.join(model_dir, "weights.hdf5")
    model.load_weights(weights_file)

    return model

def save_model(model, path):
    """Save a keras model and its weights at the given path."""

    print("Save trained model to {}.".format(path))
    model_path = os.path.join(path, "model.json")
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    weights_path = os.path.join(path, "weights.hdf5")
    model.save_weights(weights_path)


