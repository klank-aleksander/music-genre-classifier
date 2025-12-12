import json
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

# --- FIX DLA MACA (Wymuszenie CPU) ---
# Jeśli chcesz spróbować GPU, zakomentuj te dwie linie, ale zalecam zostawić
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- KONFIGURACJA ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

DATA_PATH = os.path.join(project_root, "data/processed/data.json")
MODEL_SAVE_PATH = os.path.join(project_root, "models/music_genre_model.keras")

TEST_SIZE = 0.25
VALIDATION_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 30  # CNN uczy się szybciej, 30 epok powinno wystarczyć


def load_data(data_path):
    """Wczytuje dane treningowe z pliku JSON."""
    with open(data_path, "r") as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    print("Dane wczytane pomyślnie!")
    return inputs, targets


def prepare_datasets(test_size, validation_size):
    """Wczytuje dane i przygotowuje je dla CNN (dodaje 3 wymiar)."""

    # 1. Wczytaj dane
    X, y = load_data(DATA_PATH)

    # 2. Podział: Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # 3. Podział: Train / Validation
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # --- KLUCZOWA POPRAWKA DLA CNN ---
    # Dodajemy "kanał" na końcu.
    # Z kształtu (100, 130, 13) robimy (100, 130, 13, 1)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape):
    """Buduje architekturę sieci CNN."""

    model = tf.keras.Sequential()

    # POPRAWKA WARNINGU: Jawna warstwa Input
    model.add(tf.keras.layers.Input(shape=input_shape))

    # 1. Blok konwolucyjny
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2. Blok konwolucyjny
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3. Blok konwolucyjny
    model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 4. Spłaszczenie i klasyfikacja
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    # Wyjście (10 gatunków)
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Kompilacja
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def plot_history(history):
    """Rysuje wykresy uczenia."""
    fig, axs = plt.subplots(2, figsize=(10, 8))

    # Accuracy
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="val accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # Error
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="val error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error evaluation")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 0. Katalog na modele
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # 1. Przygotuj dane (Teraz zwraca poprawne kształty 4D)
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(TEST_SIZE, VALIDATION_SIZE)

    # 2. Zbuduj model
    # input_shape bierze teraz (130, 13, 1) z X_train
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # Wyświetl architekturę (powinno być widać Output Shape z '1' na końcu)
    model.summary()

    # 3. Trenuj
    print("\nRozpoczynam trening modelu CNN...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_validation, y_validation),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)

    # 4. Ewaluacja
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%")

    # 5. Zapisz
    model.save(MODEL_SAVE_PATH)
    print(f"Model zapisany w: {MODEL_SAVE_PATH}")

    plot_history(history)