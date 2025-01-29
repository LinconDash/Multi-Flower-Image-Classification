FROM python:3.10-slim

COPY requirements.txt .
COPY models/ models/

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY . .

EXPOSE 8080

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES="-1"

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]