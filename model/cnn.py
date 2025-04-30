from math import ceil, floor
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers # type: ignore
from sklearn.model_selection import train_test_split

NUM_CLASSES = 5  # Number of emotion labels
emotion_to_number = {
        'Anger': 0,
        'Happy': 1,
        'Neutral': 2,
        'Sad': 3,
        'Surprise': 4,
}

def print_with_dashes(string=""):
    total_length = 60

    dash_count_left = floor((total_length - len(string)) / 2)
    dash_count_right = ceil((total_length - len(string)) / 2)
    print(f"{'-' * dash_count_left}{string}{'-' * dash_count_right}")

def create_model(input_shape, num_classes):
    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=input_shape))

    # Convolutional Layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the 2D feature maps
    model.add(layers.Flatten())
    # Dense Layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def load_audio(folder_path=None):
    data_list = []
    label_list = []
    data_name = "data_cache.npy"
    label_name = "label_cache.npy"
    loaded_sets = 0

    if folder_path is None:
        folder_path = "train_data"

    print_with_dashes("LOADING DATA FROM NPY CACHE")
    for subdir, dirs, files in os.walk(folder_path):
        print(f"Checking '{os.path.basename(subdir)}'")
        if len(dirs) == 0 and len(files) != 2:
            raise FileNotFoundError(f"Expected 2 files but found {len(files)} in '{subdir}'")

        if len(dirs) == 0:
            data_path = os.path.join(subdir, data_name)
            label_path = os.path.join(subdir, label_name)
            # Ensure both data and label npy file exist
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Expected a file named {data_name}, but none were found in '{subdir}'")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Expected a file named {label_name}, but none were found in '{subdir}'")
            # Load cache
            data = np.load(data_path)
            labels = np.load(label_path)
            print(f"+ Loaded dataset '{os.path.basename(subdir)}' with shape: {data.shape}")

            data_list.append(data)
            label_list.append(labels)
            loaded_sets += 1
    all_data = np.concatenate(data_list, axis=0)
    all_labels = np.concatenate(label_list, axis=0)

    print_with_dashes("RESULTS")
    print(f"Successfully loaded {loaded_sets} cached datasets")
    print(f"All data: {all_data.shape}")
    print(f"All labels: {all_labels.shape}")

    # Add channel dimension (for CNN input)
    all_data = np.expand_dims(all_data, axis=-1)  # Shape: (num_samples, time_frames, mel_bins, 1)
    # Convert labels to categorical (one-hot encoding)
    all_labels = tf.keras.utils.to_categorical(all_labels, num_classes=NUM_CLASSES)

    # Split into training and validation sets
    print_with_dashes("TRAINING DATA")
    X_train, X_val, y_train, y_val = train_test_split(all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print_with_dashes()
    return X_train, X_val, y_train, y_val
