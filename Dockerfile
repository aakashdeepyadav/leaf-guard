FROM python:3.11-slim

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY src/ src/
COPY app/ app/
COPY models/ models/
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]
