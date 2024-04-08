import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import to_categorical

def preprocess_audio(audio_file_path, max_pad_len=400):
    # Load audio
    y, sr = librosa.load(audio_file_path, sr=16000)

    # Normalize volume
    y /= np.max(np.abs(y))

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # If the audio file is shorter than the max_pad_len, pad it with zeros
    if (mfccs.shape[1] < max_pad_len):
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # If the audio file is longer than the max_pad_len, truncate it
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs

def preprocess_audio_folder(folder_path):
    mfcc_feature_list = []
    labels_list = []

    for folder_name in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_path_full):
            for filename in os.listdir(folder_path_full):
                if filename.endswith(".wav"):
                    audio_path = os.path.join(folder_path_full, filename)
                    mfcc_features = preprocess_audio(audio_path)
                    mfcc_feature_list.append(mfcc_features)
                    labels_list.append(folder_name)

    return mfcc_feature_list, labels_list

if __name__ == "__main__":
    audio_folder = r"C:\Users\apaar\Downloads\audio_files"  # Replace with the actual folder path
    max_pad_len = 400
    # Preprocess audio files in the folder
    mfcc_features_list, labels_list = preprocess_audio_folder(audio_folder)

    # Convert labels to numerical values
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels_list)
    num_classes = len(label_encoder.classes_)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        mfcc_features_list, labels_encoded, test_size=0.2, random_state=42
    )

    # Build a simple feedforward neural network
    model = Sequential()
    model.add(Flatten(input_shape=(13, max_pad_len)))  # Add this line
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        np.array(X_train),
        np.array(y_train),
        epochs=1000,
        batch_size=32,
        validation_data=(np.array(X_test), np.array(y_test)),
        verbose=1,
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(np.array(X_test), np.array(y_test))
    print(f"Test accuracy: {test_accuracy:.4f}")

    # You can now use the trained model for prediction
    # Remember to save the model for future use
