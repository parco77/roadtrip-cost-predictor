# Two-stage build: Node compiles the React app, Python serves it alongside the API.
# The frontend never ships in the final image — only its built output.

FROM node:20-slim AS frontend
WORKDIR /build
# Copy manifests first so `npm ci` is cached unless dependencies actually change.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
# vite.config.js writes to ../static, which is outside this WORKDIR — point it here instead.
RUN npm run build -- --outDir dist --emptyOutDir


FROM python:3.11-slim AS runtime
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the server actually needs at runtime. The notebook, the raw CSV, the scripts and
# the frontend source are all build-time or documentation artifacts.
#
# roadtrip_features.py and models/pipeline.joblib are both read at module scope — app.py imports
# the first and joblib.loads the second before it serves a single request — so omitting either
# makes the container exit on start. That failure is invisible behind render.yaml's health check,
# which just reports a deploy that never goes live, so it is worth stating why they are here.
COPY app.py roadtrip_features.py ./
COPY data/india_cities.csv data/fuel_prices.csv ./data/
COPY models/model.joblib models/pipeline.joblib ./models/
COPY --from=frontend /build/dist ./static

# Most hosts inject $PORT. Default to 8000 for plain `docker run`.
ENV PORT=8000
EXPOSE 8000

# Single worker on purpose: the model and the 3,739-city table are loaded per process, and the
# OSRM distance cache is in-process, so extra workers multiply memory and cold routing calls.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
