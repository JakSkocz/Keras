import numpy as np
import matplotlib.pyplot as plt
from keras.api.models import load_model
from keras.api.preprocessing.image import load_img, img_to_array

# Ładowanie wytrenowanego modelu
model = load_model('my_model30.keras')


def classify_image(image_path):
    # Załadowanie obrazu
    img = load_img(image_path, target_size=(28, 28), color_mode='grayscale')  # Dostosowanie rozmiar obrazu
    img_array = img_to_array(img)  # Konwersja do tablicy
    img_array = img_array.reshape((1, 784))  # Przekształć na wektor 784-elementowy
    img_array = img_array.astype('float32') / 255  # Normalizacja

    # Dokonanie predykcji
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)  # Klasa z najwyższym prawdopodobieństwem

    return predicted_class[0]


# Ścieżka do obrazu
image_path = r'mnist1.png'

predicted_class = classify_image(image_path)
print(f'Predykowana klasa: {predicted_class}')

# Wyświetlenie obrazu
img = load_img(image_path, color_mode='grayscale')
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.title(f'Predykowana klasa: {predicted_class}')
plt.show()
