"""
Moduł odpowiedzialny za trenowanie sieci neuronowej.
Wczytuje przetworzone dane, buduje model CNN i zapisuje wagi oraz wykresy.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- KONFIGURACJA ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

DATA_PATH = os.path.join(project_root, "data/processed/data.json")
MODEL_SAVE_PATH = os.path.join(project_root, "models/music_genre_model.keras")
FIGURES_PATH = os.path.join(project_root, "reports/figures")

TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
BATCH_SIZE = 32
EPOCHS = 30


def load_data(data_path):
    """Wczytuje dane treningowe i mapowanie etykiet z pliku JSON."""
    with open(data_path, "r", encoding='utf-8') as fp:
        data = json.load(fp)

    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    mapping = data.get("mapping", [])  # Pobieramy nazwy gatunków

    print("Dane wczytane pomyślnie!")
    return inputs, targets, mapping


def prepare_datasets(test_size, validation_size):
    """Wczytuje dane i przygotowuje je dla CNN (ze stratyfikacją)."""

    X, y, mapping = load_data(DATA_PATH)

    # Dodano stratify=y -> gwarantuje równą liczbę gatunków w obu zbiorach
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size, stratify=y_train
    )

    # dodajemy wymiar kanału (dla CNN)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test, mapping


def build_model(input_shape):
    """Buduje architekturę sieci CNN."""

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # 1. Blok
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 2. Blok
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # 3. Blok
    model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    #Wyjście
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def save_plots(history, save_dir):
    """Zapisuje wykresy Accuracy i Loss do pliku PNG."""
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

    # Zapis do pliku
    save_path = os.path.join(save_dir, "training_history.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)  # Zamknij, żeby zwolnić pamięć
    print(f"Wykres uczenia zapisano w: {save_path}")


def save_confusion_matrix(model, X_test, y_test, mapping, save_dir):
    """Generuje i zapisuje macierz pomyłek (bez użycia seaborn)."""

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Obliczenie macierzy
    cm = confusion_matrix(y_test, y_pred_classes)

    # Rysowanie
    fig, ax = plt.subplots(figsize=(10, 10))

    # Wyświetlanie macierzy
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mapping)

    # Rysowanie
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)

    ax.set_title('Macierz Pomyłek (Confusion Matrix)')

    # Zapis
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Macierz pomyłek zapisano w: {save_path}")


if __name__ == "__main__":
    # Katalogi
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)  # Tworzymy folder na raporty

    # Dane
    # mapping (lista gatunków) do podpisania osi na wykresie
    X_train, X_validation, X_test, y_train, y_validation, y_test, mapping = \
        prepare_datasets(TEST_SIZE, VALIDATION_SIZE)

    # Model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # Trening
    print("\nRozpoczynam trening modelu CNN...")
    history = model.fit(X_train, y_train,
                        validation_data=(X_validation, y_validation),
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)

    # Ewaluacja liczbowa
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%")

    # Zapis modelu
    model.save(MODEL_SAVE_PATH)
    print(f"Model zapisany w: {MODEL_SAVE_PATH}")

    # Generowanie raportów graficznych
    print("\nGenerowanie wykresów...")
    save_plots(history, FIGURES_PATH)

    # Macierz pomyłek tylko jeśli mamy mapping gatunków
    if mapping:
        save_confusion_matrix(model, X_test, y_test, mapping, FIGURES_PATH)
    else:
        # Fallback gdyby mapping był pusty
        print("Brak mapowania gatunków w JSON, pomijam opisy osi macierzy.")
        dummy_mapping = [str(i) for i in range(10)]
        save_confusion_matrix(model, X_test, y_test, dummy_mapping, FIGURES_PATH)