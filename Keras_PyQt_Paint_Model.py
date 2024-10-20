# Implementacja obsługi ładowania i predykcji modelu
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Wyłącza informacje i ostrzeżenia
from keras.api.models import model_from_json
from keras.api.models import load_model
import cv2
import numpy as np

from PyQt6.QtGui import QImage


def qimage_to_array(image: QImage):
    """
    Funkcja konwertująca obiekt QImage do numpy array
    """
    image = image.convertToFormat(QImage.Format.Format_Grayscale8)
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    numpy_array = np.array(ptr).reshape(image.height(), image.width(), 1)

    # wykorzystanie bibloteki OpenCV do wyświetlenia obrazu po konwersji
    #cv2.imshow('Check if the function works!', numpy_array)
    return numpy_array


def predict(image: QImage, model):
    """
    Funkcja wykorzystująca załadowany model sieci neuronowej do predykcji znaku na obrazie

    Należy dodać w niej odpowiedni kod do obsługi załadowanego modelu
    """
    numpy_array = qimage_to_array(image)
    print("Predicted array shape before resizing:", numpy_array.shape)

    # Zmiana rozmiaru na 28x28
    numpy_array = cv2.resize(numpy_array, (28, 28))
    print("Resized array shape:", numpy_array.shape)

    # Normalizacja obrazu
    numpy_array = numpy_array.astype(np.float32) / 255.0

    # Spłaszczenie obrazu do wektora (1, 784)
    numpy_array = numpy_array.reshape(1, 28 * 28)
    print("Final input shape for prediction (flattened):", numpy_array.shape)

    try:
        # Wykonanie predykcji
        predicted_class = model.predict(numpy_array)
        print("Prediction output:", predicted_class)

        predicted_label = np.argmax(predicted_class)
        print("Predicted label:", predicted_label)

        return predicted_label
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

    print("Predicted label:", predicted_label)
    return predicted_label


def get_model():
    try:
        model = load_model('my_model100.keras')
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None