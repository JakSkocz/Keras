# Ładowanie potrzebych modułów
from keras.api.models import Sequential, load_model
from keras.api.layers import Dense, Dropout
from keras.api.utils import to_categorical
from matplotlib import pyplot as plt
from keras.api.datasets import mnist

# Wczytywanie danych
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Splaszczenie obrazów z 28 * 28 pikseli do 784 elementowego wektora
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

# Normalizacja danych o wartościach od 0 do 255 do wartości od 0 do 1
X_train = X_train / 255
X_test = X_test / 255

# Pobranie i stworzenie listy klas dla danych
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Wyciągnięcie liczby klas
num_classes = y_test.shape[1]

# Tworzenie modelu sieci
model = Sequential()

# Dodanie pierwszej warstwy gęstej z większą liczbą neuronów
model.add(Dense(256, input_dim=num_pixels, activation='relu'))
model.add(Dropout(0.3))
# Dodanie drugiej warstwy gęstej
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
# Dodanie warstwy wyjściowej
model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Uczenie modelu danymi
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=200, verbose=1)

# Testowanie modelu
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

# Zapis modelu do pliku
model.save('my_model100.keras')  # Zapisuje cały model w formacie Keras

# Odczyt modelu z pliku
loaded_model = load_model('my_model100.keras')  # Ładowanie modelu z formatu Keras

# Testowanie załadowanego modelu
loaded_scores = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Loaded model Baseline Error: %.2f%%" % (100 - loaded_scores[1] * 100))

# Wyświetlenie wykresu przedstawiającego historię uczenia sieci
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
