#!/bin/bash

mgc() {
    echo "###############################"
    echo "# Music Genre Classifier 2025 #"
    echo "###############################"
}

if [[ $EUID -ne 0 ]]; then
  echo "Ten skrypt musi być uruchomiony jako root."
  exit 1
fi

cd "$(dirname "$0")" || exit 1

mgc

if ! docker info > /dev/null 2>&1; then
    echo "Przed uruchomieniem programu należy zainstalować Docker i Docker-Compose."
    read -p "Naciśnij Enter, aby kontynuować..."
    exit 1
fi

echo "Trwa uruchamianie programu..."

if ! docker image inspect music-genre-classifier-app:latest > /dev/null 2>&1; then
    echo "Pierwsze uruchomienie może potrwać kilka minut"
    echo "NIE ZAMYKAJ TEGO OKNA!"
fi

cleanup() {
    docker compose down
    exit 0
}

trap cleanup EXIT SIGINT SIGTERM SIGHUP

docker compose up -d

clear
sleep 3
mgc

echo "Otwórz przeglądarkę i przejdź pod adres: http://127.0.0.1:8080"

while true; do
    sleep 60
done

