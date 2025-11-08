FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml README.md model-card.md /app/
RUN pip install --no-cache-dir .
COPY src /app/src
COPY config /app/config
COPY assets /app/assets

ENTRYPOINT ["csc-serve"]
CMD ["--help"]
