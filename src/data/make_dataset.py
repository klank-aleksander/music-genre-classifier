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
    Ekstrahuje cechy MFCC wymuszając ALFABETYCZNĄ kolejność gatunków.
    """

    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # 1. Pobierz listę folderów i POSORTUJ JĄ ALFABETYCZNIE
    # wymuszamy kolejność ["blues", "classical", ...]
    genres = sorted(
        [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')])

    print(f"Ustalona kolejność gatunków: {genres}")
    data["mapping"] = genres

    # 2. Iteruj po posortowanych gatunkach
    for i, genre in enumerate(genres):
        print(f"\nPrzetwarzanie: {genre} (Klasa: {i})")

        genre_path = os.path.join(dataset_path, genre)

        # Iteruj po plikach wewnątrz gatunku
        for f in os.listdir(genre_path):
            file_path = os.path.join(genre_path, f)

            if not f.endswith('.wav'):
                continue

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                for s in range(num_segments):
                    start_sample = samples_per_segment * s
                    finish_sample = start_sample + samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i)  # Teraz 'i' odpowiada indeksowi na liście alfabetycznej

            except Exception as e:
                print(f"Błąd przy pliku {file_path}: {e}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

    print(f"\nSukces! Dane zapisane w {json_path}")

if __name__ == "__main__":
    import math

    output_dir = os.path.dirname(JSON_PATH)
    os.makedirs(output_dir, exist_ok=True)

    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)