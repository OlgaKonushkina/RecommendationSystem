FROM python:3.13-slim

WORKDIR /app/service

# Копируем и устанавливаем зависимости
COPY service/requirements-minimal.txt .
RUN pip install --no-cache-dir -r requirements-minimal.txt

# Копируем код сервиса
COPY service/ .

# Копируем модель
COPY models/catboost_ranker.bin ../models/

# Запускаем приложение
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]