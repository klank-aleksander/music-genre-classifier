"""
Moduł obsługujący logikę predykcji.
Zawiera klasę GenreClassifier, która ładuje model i klasyfikuje pliki audio (całe utwory).
"""
import os
import math
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# --- KONFIGURACJA ---
SAMPLE_RATE = 22050
TRAINING_DURATION = 30  # Na takich plikach trenowaliśmy
SEGMENT_DURATION = 3    # Długość jednego wycinka w sekundach (1/10 z 30s)

# Obliczamy ile próbek ma jeden segment (kluczowe dla modelu)
SAMPLES_PER_SEGMENT = int(SAMPLE_RATE * SEGMENT_DURATION)
EXPECTED_MFCC_VECTORS = 130  # Tyle wektorów czasowych oczekuje model (dla 3s)

# Lista gatunków w kolejności alfabetycznej
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


class GenreClassifier:
    """
    Wrapper na model TensorFlow.
    Zajmuje się ładowaniem modelu, preprocessingiem audio i wnioskowaniem.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, model_path):
        """Ładuje model przy starcie aplikacji."""
        self.model = load_model(model_path, compile=False)
        print(f"Model załadowany z: {model_path}")

    def _preprocess_audio(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Przerabia CAŁY plik audio na zestaw macierzy MFCC.
        Dzieli utwór na segmenty po 3 sekundy.
        """
        # pylint: disable=broad-exception-caught
        try:
            # 1. Wczytaj CAŁY plik audio (bez limitu duration)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Oblicz ile pełnych 3-sekundowych segmentów mieści się w utworze
            num_segments = len(signal) // SAMPLES_PER_SEGMENT

            if num_segments == 0:
                print("Plik jest za krótki (mniej niż 3 sekundy).")
                return None

            processed_segments = []

            # 2. Pętla po wszystkich możliwych segmentach
            for s in range(num_segments):
                start_sample = SAMPLES_PER_SEGMENT * s
                finish_sample = start_sample + SAMPLES_PER_SEGMENT

                # Wyciągnij kawałek sygnału
                chunk = signal[start_sample:finish_sample]

                # Zrób MFCC
                mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T  # Transpozycja (czas, cechy)

                # Sprawdź czy wymiar się zgadza (musi być idealnie 130 ramek)
                if len(mfcc) == EXPECTED_MFCC_VECTORS:
                    processed_segments.append(mfcc.tolist())
                else:
                    # Czasem przez zaokrąglenia wyjdzie 129 lub 131 - takie odrzucamy dla bezpieczeństwa
                    continue

            # Jeśli po filtracji nic nie zostało
            if not processed_segments:
                return None

            # 3. Zamień na numpy array i dodaj wymiar kanału (dla CNN)
            # Wynik: (ilość_kawałków, 130, 13, 1)
            X = np.array(processed_segments)
            X = X[..., np.newaxis]

            return X

        except Exception as e:
            print(f"Błąd przetwarzania pliku: {e}")
            return None

    def predict(self, file_path):
        """Główna funkcja: Audio -> Klasa -> Wynik"""

        # 1. Przygotuj dane (potnij CAŁY utwór na kawałki)
        X = self._preprocess_audio(file_path)

        if X is None or len(X) == 0:
            return None

        # 2. Wykonaj predykcję dla KAŻDEGO kawałka
        # predictions to macierz (N kawałków x 10 gatunków)
        predictions = self.model.predict(X, verbose=0)

        # 3. GŁOSOWANIE (Średnia z prawdopodobieństw wszystkich kawałków)
        avg_prediction = np.mean(predictions, axis=0)

        # 4. Znajdź zwycięzcę
        predicted_index = np.argmax(avg_prediction)
        predicted_genre = GENRES[predicted_index]
        confidence = avg_prediction[predicted_index]

        return {
            "genre": predicted_genre,
            "confidence": float(confidence),
            "probabilities": dict(zip(GENRES, avg_prediction.tolist()))
        }


# --- TEST LOKALNY ---
if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_path))
    model_path = os.path.join(project_root, "models/music_genre_model.keras")

    # Przykładowy plik
    test_file = os.path.join(project_root, "data/raw/genres_original/metal/metal.00004.wav")

    classifier = GenreClassifier(model_path)

    if os.path.exists(test_file):
        print(f"\nAnalizuję plik: {test_file}...")
        result = classifier.predict(test_file)

        if result:
            print("\n--- WYNIKI ---")
            print(f"Gatunek: {result['genre'].upper()}")
            print(f"Pewność: {result['confidence'] * 100:.2f}%")