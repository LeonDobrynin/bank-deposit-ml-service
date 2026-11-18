FROM python:3.12-slim
WORKDIR /service
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG MODEL_VERSION=1.1.0
COPY app/ app/
COPY artifacts/${MODEL_VERSION}/ artifacts/current/
ENV ARTIFACT_DIR=/service/artifacts/current
USER 10001:10001
EXPOSE 8077
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8077/health')"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8077"]
