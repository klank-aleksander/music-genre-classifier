@echo off
set "script=%~dp0win.ps1"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
 "Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%script%\"'"

:: Tutaj dopiero zostanie wykonane po zamknięciu okna PS
docker-compose down
















@REM @echo off
@REM set "script=%~dp0win.ps1"
@REM
@REM powershell -NoProfile -ExecutionPolicy Bypass -Command ^
@REM  "Start-Process powershell -Verb RunAs -PassThru -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File \"%script%\"'" > "%temp%\ps_pid.txt"
@REM
@REM set /p pid=<"%temp%\ps_pid.txt"
@REM del "%temp%\ps_pid.txt"
@REM
@REM
@REM timeout /t 1 >nul
@REM tasklist /fi "PID eq %pid%" | findstr /i "%pid%" >nul
@REM if %errorlevel%==0 goto waitLoop
@REM
@REM docker-compose down


@REM echo ###############################
@REM echo # Music Genre Classifier 2025 #
@REM echo ###############################
@REM
@REM docker --version >nul 2>&1
@REM if %errorlevel% neq 0 (
@REM     echo Przed uruchomieniem programu należy zainstalować oprogramowanie Docker Desktop
@REM     exit /b 1
@REM )
@REM
@REM
@REM docker image inspect music-genre-classlifier-app:latest >nul 2>&1
@REM if %errorlevel% neq 0 (
@REM     echo Rozpoczynianie instalacji.
@REM     echo Pierwsze uruchomienie może potrwać kilka minut
@REM     echo NIE ZAMYKAJ TEGO OKNA!
@REM     exit /b 1
@REM )
@REM
@REM docker-compose.exe up -d
