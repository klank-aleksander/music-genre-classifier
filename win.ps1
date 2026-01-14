function mgc {
    Write-Host "###############################"
    Write-Host "# Music Genre Classifier 2025 #"
    Write-Host "###############################"
}

Set-Location $PSScriptRoot

mgc

docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Przed uruchomieniem programu nale¿y zainstalowaæ oprogramowanie Docker Desktop."
    Pause
    exit 1
}

Write-Host "Trwa uruchamianie programu..."

docker image inspect music-genre-classifier-app:latest *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host  "Pierwsze uruchomienie mo¿e potrwaæ kilka minut"
    Write-Host  "NIE ZAMYKAJ TEGO OKNA!"
}

docker compose up -d
clear
Start-Sleep -Seconds 3
mgc
Start-Process "http://127.0.0.1:8080"

Write-Host "Program dostêpny jest pod adresem http://127.0.0.1:8080"

while ($true) {
    Start-Sleep -Seconds 60
}