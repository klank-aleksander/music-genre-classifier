import streamlit as st
import os
import sys
import pandas as pd
from tempfile import NamedTemporaryFile

# Dodaj ścieżkę do folderu 'src', aby móc importować własne moduły
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.predict import GenreClassifier, GENRES

st.title("Music Genre Classifier")

# --- KONFIGURACJA ---
# Ustal ścieżkę do modelu względem TEGO pliku
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/music_genre_model.keras")

# --- ŁADOWANIE MODELU (z cache) ---
@st.cache_resource
def load_classifier(model_path):
    """Ładuje model i cachuje go, aby nie ładować go przy każdej interakcji."""
    if not os.path.exists(model_path):
        st.error(f"Nie znaleziono modelu pod ścieżką: {model_path}")
        st.error("Upewnij się, że model 'music_genre_model.keras' znajduje się w folderze 'models'.")
        return None
    classifier = GenreClassifier(model_path)
    return classifier

classifier = load_classifier(MODEL_PATH)

# --- UI APLIKACJI ---
uploaded_file = st.file_uploader("Wrzuć plik audio",
                                 type=[
                                     "wav", "mp3", "ogg", "flac", "aiff", "aif"
                                 ])

if classifier and uploaded_file is not None:
    ext = uploaded_file.name.split(".")[-1].lower()
    mime_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "aiff": "audio/aiff",
        "aif": "audio/aiff",
    }
    mime = mime_map.get(ext, "audio/wav")

    st.audio(uploaded_file, format=mime)

    # Przycisk do uruchomienia analizy
    if st.button("Klasyfikuj gatunek"):
        with st.spinner("Analizuję utwór... To może chwilę potrwać."):
            # Zapisz tymczasowo plik, aby `librosa` mógł go odczytać z ścieżki
            with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name

            try:
                # Wykonaj predykcję
                result = classifier.predict(temp_path)

                if result:
                    st.success(f"Oszacowany gatunek: **{result['genre'].upper()}**")
                    st.metric(label="Pewność", value=f"{result['confidence'] * 100:.2f}%")

                    # Przygotuj dane do wykresu
                    probabilities = result['probabilities']
                    df = pd.DataFrame(
                        probabilities.values(),
                        index=probabilities.keys(),
                        columns=["Prawdopodobieństwo"]
                    )

                    # Wykres słupkowy
                    st.bar_chart(df)
                else:
                    st.error("Nie udało się przetworzyć pliku audio.")

            finally:
                # Posprzątaj po sobie
                if os.path.exists(temp_path):
                    os.remove(temp_path)
