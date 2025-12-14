import os
import numpy as np
import librosa
import tensorflow as tf

# --- KONFIGURACJA ---
SAMPLE_RATE = 22050
DURATION = 30  # Tyle sekund analizujemy
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
NUM_SEGMENTS = 10  # Na tyle części dzieliliśmy utwór w treningu

# Lista gatunków w kolejności alfabetycznej
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


class GenreClassifier:
    def __init__(self, model_path):
        """Ładuje model przy starcie aplikacji."""
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model załadowany z: {model_path}")

    def _preprocess_audio(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Przerabia plik audio na zestaw macierzy MFCC gotowych dla modelu CNN.
        Zwraca tablicę o kształcie: (liczba_segmentów, 130, 13, 1)
        """
        try:
            # 1. Wczytaj audio
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # Jeśli plik jest krótszy niż 30s, zapętl go lub zostaw
            # Dla uproszczenia bierzemy po prostu tyle ile jest, ale ucinamy do 30s max
            if len(signal) > SAMPLES_PER_TRACK:
                signal = signal[:SAMPLES_PER_TRACK]

            # 2. Parametry segmentacji
            samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
            num_mfcc_vectors_per_segment = 130  # Oczekiwana długość czasowa przez model

            processed_segments = []

            # 3. Pętla po segmentach (tnij utwór na kawałki po 3 sekundy)
            for s in range(NUM_SEGMENTS):
                start_sample = samples_per_segment * s
                finish_sample = start_sample + samples_per_segment

                # Wyciągnij kawałek sygnału
                chunk = signal[start_sample:finish_sample]

                # Zabezpieczenie dla bardzo krótkich plików
                if len(chunk) < samples_per_segment:
                    continue

                # Zrób MFCC
                mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfcc = mfcc.T  # Transpozycja (czas, cechy)

                # Sprawdź czy wymiar się zgadza (czasem jest +/- 1 ramka różnicy przez zaokrąglenia)
                if len(mfcc) == num_mfcc_vectors_per_segment:
                    processed_segments.append(mfcc.tolist())

            # 4. Zamień na numpy array i dodaj wymiar kanału (dla CNN)
            # Wynik: (ilość_kawałków, 130, 13, 1)
            X = np.array(processed_segments)
            X = X[..., np.newaxis]

            return X

        except Exception as e:
            print(f"Błąd przetwarzania pliku: {e}")
            return None

    def predict(self, file_path):
        """Główna funkcja: Audio -> Klasa -> Wynik"""

        # 1. Przygotuj dane (potnij na kawałki)
        X = self._preprocess_audio(file_path)

        if X is None or len(X) == 0:
            return None

        # 2. Wykonaj predykcję dla KAŻDEGO kawałka
        # predictions to macierz (10 kawałków x 10 gatunków)
        predictions = self.model.predict(X, verbose=0)

        # 3. GŁOSOWANIE (Średnia z prawdopodobieństw wszystkich kawałków)
        # Np. Kawałek 1 mówi: 90% Rock. Kawałek 2 mówi: 80% Rock. Średnia = Rock.
        avg_prediction = np.mean(predictions, axis=0)

        # 4. Znajdź zwycięzcę
        predicted_index = np.argmax(avg_prediction)
        predicted_genre = GENRES[predicted_index]
        confidence = avg_prediction[predicted_index]

        # 5. Zwróć wynik i pełny rozkład (do wykresu)
        return {
            "genre": predicted_genre,
            "confidence": float(confidence),
            "probabilities": dict(zip(GENRES, avg_prediction.tolist()))
        }


# --- TEST LOKALNY (Tylko gdy uruchamiasz ten plik bezpośrednio) ---
if __name__ == "__main__":
    # Ustal ścieżki
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_path))
    model_path = os.path.join(project_root, "models/music_genre_model.keras")

    # Przetestuj na jakimś pliku z datasetu (np. Metal)
    test_file = os.path.join(project_root, "data/raw/genres_original/metal/metal.00004.wav")

    # Inicjalizacja
    classifier = GenreClassifier(model_path)

    if os.path.exists(test_file):
        print(f"\nAnalizuję plik: {test_file}...")
        result = classifier.predict(test_file)

        print("\n--- WYNIKI ---")
        print(f"Gatunek: {result['genre'].upper()}")
        print(f"Pewność: {result['confidence'] * 100:.2f}%")
        print("Rozkład:", result['probabilities'])
    else:
        print("Nie znaleziono pliku testowego. Zmień ścieżkę w bloku 'if __name__'.")