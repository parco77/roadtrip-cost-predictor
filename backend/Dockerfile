# The API alone. The React UI builds and deploys on its own host, so there is no Node stage
# here and nothing to copy between stages.
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# `COPY . .` rather than a list of files. This folder holds only what the server runs -
# app.py, roadtrip_features.py, the two CSVs and the two model bundles - so an explicit
# list would be a second inventory to keep in step, and the earlier one silently omitted
# two files that are read at import, which a health check reports as a deploy that never
# goes live. .dockerignore carries the exclusions instead.
COPY . .

# Most hosts inject $PORT. Default to 8000 for a plain `docker run`.
ENV PORT=8000
EXPOSE 8000

# One worker on purpose: both model bundles and the 3,739-city table are loaded per process,
# and the OSRM distance cache is in-process, so a second worker doubles memory and re-fetches
# every route it has not seen.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
