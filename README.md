# Music Genre Classifier 🎵

> Aplikacja wykorzystująca uczenie maszynowe (Machine Learning) do klasyfikacji gatunków muzycznych na podstawie plików audio, oparta na Konwolucyjnych Sieciach Neuronowych (CNN) i analizie cech MFCC.

![Version](https://img.shields.io/badge/version-0.0.1-blue)
![Python](https://img.shields.io/badge/python-3.12-yellow)
![Streamlit](https://img.shields.io/badge/streamlit-1.31-red)
![Docker](https://img.shields.io/badge/docker-available-blue)

## Spis treści
- [O projekcie](#o-projekcie)
- [Struktura projektu](#struktura-projektu)
- [Technologie](#technologie)
- [Instalacja i Uruchomienie](#instalacja-i-uruchomienie)
  - [Wymagania wstępne](#wymagania-wstępne)
  - [Metoda 1: Docker Compose (Zalecana)](#metoda-1-docker-compose-zalecana)
  - [Metoda 2: Czysty Docker](#metoda-2-czysty-docker)
  - [Metoda 3: Uruchomienie lokalne (Python)](#metoda-3-uruchomienie-lokalne-python)
- [Zbiór danych i Trenowanie](#zbiór-danych-i-trenowanie-opcjonalne)
- [Autorzy](#autorzy)

---

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

---

## Instalacja i Uruchomienie

Aplikację można uruchomić w kontenerze (zalecane) lub bezpośrednio w środowisku Python.

### Wymagania wstępne
* Zainstalowany **Git**.
* Zainstalowany **Docker** oraz **Docker Compose** (dla metod 1 i 2).
* **Python 3.12** (tylko dla metody 3).

### Pobranie kodu
Na początku sklonuj repozytorium na swój komputer:

```bash
git clone https://github.com/klank-aleksander/music-genre-classifier.git
cd music-genre-classifier
```
### Metoda 1: Docker Compose (Zalecana)
Najprostszy sposób uruchomienia. Automatycznie buduje obraz i mapuje porty.

1. Zbuduj i uruchom kontener:
    ```bash
       docker-compose up --build -d
    ```
2. Otwórz przeglądarkę pod adresem: http://localhost:8080

3. Aby zatrzymać aplikację wpisz:
    ```bash
       docker-compose down
    ```

### Metoda 2: Czysty Docker

Jeśli nie chcesz używać Compose, możesz zbudować obraz ręcznie.

1. Zbuduj obraz:

    ```bash
    docker build -t music-classifier .
    ```

2. Uruchom kontener (mapując port hosta 8080 na port kontenera 80):

    ```bash
    docker run -p 8080:80 music-classifier
    ```

3. Aplikacja dostępna pod adresem http://localhost:8080.

### Metoda 3: Uruchomienie lokalne (Python)

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



# Zbiór danych i Trenowanie (Opcjonalne)

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