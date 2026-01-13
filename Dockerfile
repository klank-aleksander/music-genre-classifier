FROM python:3.12-slim
LABEL authors="Aleksander Klank, Adam Dudkiewicz, Damian Zaleski"
LABEL description="Simple music genre classifier"
LABEL version="0.0.1"

WORKDIR /app

COPY requirements.txt ./
COPY ./src ./src
COPY ./models ./models


RUN python -m pip install --no-cache-dir -r requirements.txt -v


CMD ["python", "-m", "streamlit", "run", "./src/app/streamlit_app.py", "--server.port", "80"]

