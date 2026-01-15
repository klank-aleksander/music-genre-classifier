@echo off
set "script=%~dp0win.ps1"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%script%\"'"

docker-compose down

