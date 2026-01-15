# Music Genre Classifier 🎵

> Aplikacja wykorzystująca uczenie maszynowe (Machine Learning) do klasyfikacji gatunków muzycznych na podstawie plików audio, oparta na Konwolucyjnych Sieciach Neuronowych (CNN) i analizie cech MFCC.

![Version](https://img.shields.io/badge/version-0.0.1-blue)
![Python](https://img.shields.io/badge/python-3.12-yellow)
![Streamlit](https://img.shields.io/badge/streamlit-1.31-red)
![Docker](https://img.shields.io/badge/docker-available-blue)

## Spis treści
- [Music Genre Classifier 🎵](#music-genre-classifier-)
  - [Spis treści](#spis-treści)
  - [O projekcie](#o-projekcie)
  - [Źródło danych](#źródło-danych)
  - [Architektura systemu](#architektura-systemu)
  - [Struktura projektu](#struktura-projektu)
  - [Technologie](#technologie)
  - [Instalacja i Uruchomienie](#instalacja-i-uruchomienie)
    - [Wymagania wstępne](#wymagania-wstępne)
    - [Metoda 1: Uruchomienie gotowej wersji przeglądarkowej (najprostrza)](#metoda-1-uruchomienie-gotowej-wersji-przeglądarkowej-najprostrza)
    - [Metoda 2: Uruchomienie wersji lokalnej](#metoda-2-uruchomienie-wersji-lokalnej)
    - [Metoda 3: Uruchomienie wersji lokalnej (Dla programistów)](#metoda-3-uruchomienie-wersji-lokalnej-dla-programistów)
- [Trenowanie (Dla programistów)](#trenowanie-dla-programistów)
- [Autorzy](#autorzy)



## O projekcie
Celem projektu jest stworzenie kompletnego potoku (pipeline) MLOps, który przetwarza surowe pliki audio, trenuje model sieci neuronowej i udostępnia wyniki poprzez interfejs webowy.

**Główne funkcjonalności:**
* Obsługa wielu formatów audio (`.wav`, `.mp3`, `.flac`, `.ogg`, `.aiff`).
* Przetwarzanie sygnału w czasie rzeczywistym i wizualizacja wyników.
* Klasyfikacja do 10 gatunków: *Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock*.
* Konteneryzacja aplikacji zapewniająca łatwe wdrożenie.

## Źródło danych
Projekt wykorzystuje zbiór GTZAN Genre Collection (kaggle)
- 1000 fragmentów audio po 30 sekund, próbkowanie 22050Hz.
- 10 zbalansowanych kategorii (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).

## Architektura systemu

**Moduł danych**
- Pipeline przetwarzający surowe pliki audio na reprezentację matematyczną.
- Wykorzystanie transformacji Fouriera (STFT) do generowania Mel-spektrogramów oraz ekstrakcja cech MFCC (Mel-frequency cepstral coefficients).

**Moduł modelu**
- Trening modelu sieci neuronowej typu CNN.
- Ewaluacja modelu na zbiorze testowym i eksport wag do pliku.

**Moduł Aplikacji**
- Interfejs umożliwiający wgranie pliku przez użytkownika.
- Prezentacja wyniku predykcji wraz z confidence score dla poszczególnych gatunków.

## Struktura projektu
```text
music-genre-classifier/
├── data/                  # Dane
│   ├── raw/               # Surowy dataset GTZAN (ignorowany przez git)
│   └── processed/         # Przetworzone cechy (plik JSON)
├── models/                # Wytrenowane modele (.keras)
├── src/                   # Kod źródłowy
│   ├── app/               # Aplikacja frontendowa (Streamlit)
│   ├── data/              # Skrypty przetwarzania danych (ETL)
│   └── model/             # Logika trenowania i predykcji (CNN)
├── docker-compose.yml     # Konfiguracja Docker Compose
├── Dockerfile             # Definicja obrazu Docker
├── requirements.txt       # Zależności Pythonowe
└── README.md              # Dokumentacja projektu
```

## Technologie
* **Język:** Python 3.12
* **Machine Learning:** TensorFlow, Scikit-learn
* **Przetwarzanie Audio:** Librosa, NumPy
* **Wizualizacja:** Matplotlib
* **Web Framework:** Streamlit
* **DevOps:** Docker, Docker Compose


## Instalacja i Uruchomienie
### Wymagania wstępne
  * Przeglądarka internetowa
  
      Dodatkowo dla metody 1:


  * Na systemie **Windows** zainstalowany i uruchomiony **Docker Desktop**
  * Na systemie **GNU/Linux** zainstalowany i uruchomiony **Docker** oraz **Docker Compose**

### Metoda 1: Uruchomienie gotowej wersji przeglądarkowej (najprostrza)
  Gotowa wersja aplikacji dostępna jest pod tym [linkiem](https://music-genre-classifier-2wa2xppjgrts8ehfggpyfe.streamlit.app/)

### Metoda 2: Uruchomienie wersji lokalnej
  1. Pobierz i wypakuj archiwum z tego [linku](https://github.com/klank-aleksander/music-genre-classifier/archive/refs/heads/main.zip])
  2. Przejdź do folderu zawierającego pliki programu i uruchom odpowiedni plik wykonywalny:
      * **START_WINDOWS.vbs** dla systemu **Windows**
      * **START_LINUX.sh** dla systemu **GNU/Linux**
  3. Postępuj zgodnie z poleceniami wyświetlanymi w okienku.

### Metoda 3: Uruchomienie wersji lokalnej (Dla programistów)

Do prac deweloperskich bez użycia wirtualizacji.

1. Utwórz wirtualne środowisko:

    ```bash
    python -m venv .venv
    ```

2. Aktywuj wirtualne środowisko:

    - Windows:
        ```bash
        .venv\Scripts\activate
        ```
    - Mac/Linux:
        ```bash
        source .venv/bin/activate
        ```
  
3. Zainstaluj zależności:
    ```bash
    pip install -r requirements.txt
    ```

4. Uruchom aplikację Streamlit:
    ````bash
    streamlit run src/app/streamlit_app.py
    ````
   
5. Aplikacja otworzy się zazwyczaj pod adresem http://localhost:8501.


# Trenowanie (Dla programistów)

Projekt zawiera już wytrenowany model w katalogu models/. Jeśli jednak chcesz przeprowadzić trening od zera:
1. Pobierz [Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) z serwisu Kaggle.
2. Rozpakuj zawartość do folderu data/raw/genres_original/.
3. Uruchom skrypt przetwarzający dane (ekstrakcja MFCC):
    ```bash
    python src/data/make_dataset.py
    ```
4. Uruchom skrypt trenujący sieć neuronową:
    ```bash
    python src/model/train_model.py
    ```

# Autorzy
Aleksander Klank

Adam Dudkiewicz

Damian Zaleski