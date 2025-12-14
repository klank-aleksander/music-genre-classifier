import os
import json
import librosa
import numpy as np

# --- KONFIGURACJA ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

DATASET_PATH = os.path.join(project_root, "data/raw/genres_original")
JSON_PATH = os.path.join(project_root,"data/processed/data.json")
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # w sekundach
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """
    Ekstrahuje cechy MFCC z datasetu muzycznego i zapisuje je do pliku JSON.

    :param dataset_path: Ścieżka do folderu z danymi (zawiera podfoldery gatunków)
    :param json_path: Ścieżka wynikowa dla pliku JSON
    :param n_mfcc: Liczba współczynników MFCC do ekstrakcji
    :param n_fft: Długość okna FFT
    :param hop_length: Przesunięcie okna
    :param num_segments: Na ile części podzielić każdy utwór (zwiększa ilość danych treningowych)
    """

    # Słownik do przechowywania danych
    data = {
        "mapping": [],  # Np. ["blues", "classical", ...]
        "mfcc": [],  # Cechy (inputy do modelu)
        "labels": []  # Etykiety (targety, np. 0 dla bluesa)
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    print(f"Rozpoczynam przetwarzanie danych z: {dataset_path}")

    # Iteracja przez wszystkie podfoldery (gatunki)
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Pomiń główny folder datasetu, interesują nas podfoldery
        if dirpath is not dataset_path:

            # Zapisz nazwę gatunku (np. 'blues') wyciągniętą ze ścieżki
            semantic_label = dirpath.split("/")[-1]  # Dla Windows może być "\\" zamiast "/"
            if os.name == 'nt':  # Fix dla Windowsa
                semantic_label = dirpath.split("\\")[-1]

            data["mapping"].append(semantic_label)
            print(f"\nPrzetwarzanie: {semantic_label}")

            # Iteracja przez pliki audio w gatunku
            for f in filenames:
                file_path = os.path.join(dirpath, f)

                # Ignoruj pliki, które nie są audio (np. .DS_Store)
                if not f.endswith('.wav'):
                    continue

                try:
                    # Ładowanie pliku audio
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Dzielenie utworu na segmenty (żeby mieć więcej danych treningowych)
                    for s in range(num_segments):
                        start_sample = samples_per_segment * s
                        finish_sample = start_sample + samples_per_segment

                        # Ekstrakcja MFCC
                        mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                    sr=sr,
                                                    n_fft=n_fft,
                                                    n_mfcc=n_mfcc,
                                                    hop_length=hop_length)
                        mfcc = mfcc.T

                        # Zapisz tylko segmenty o oczekiwanej długości
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i - 1)  # i-1 bo os.walk na początku zwraca root

                except Exception as e:
                    print(f"Błąd przy pliku {file_path}: {e}")

    # Zapis do pliku JSON
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    print(f"\nSukces! Dane zapisane w {json_path}")


if __name__ == "__main__":
    import math

    output_dir = os.path.dirname(JSON_PATH)
    os.makedirs(output_dir, exist_ok=True)

    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)