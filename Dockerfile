FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN --mount=type=secret,id=pip_ca_bundle,target=/etc/ssl/certs/ca-certificates.crt \
    python -m pip install --no-cache-dir -r requirements.txt

RUN --mount=type=bind,source=.,target=/build-context \
    --mount=type=secret,id=pip_ca_bundle,target=/etc/ssl/certs/ca-certificates.crt \
    mkdir /tmp/meteorite \
    && cp -r /build-context/pyproject.toml /build-context/README.md /build-context/src /tmp/meteorite/ \
    && python -m pip install --no-cache-dir --no-deps /tmp/meteorite \
    && rm -r /tmp/meteorite \
    && useradd --create-home --uid 10001 meteorite \
    && install -d -o meteorite -g meteorite /app/logs /mlflow

USER meteorite
EXPOSE 8000
CMD ["uvicorn", "meteorite.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
