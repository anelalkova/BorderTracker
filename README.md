# Workers (Border Tracker)

Active code for cameras, YOLO snapshots, scrapers, and ML lives in **`CarDetector/`**, including:

- `Dockerfile` and `requirements.txt` used by `deployment/docker-compose.yml`
- `border_crossings.py` — PostgreSQL settings via **`DB_HOST`**, **`DB_PORT`**, **`DB_NAME`**, **`DB_USER`**, **`DB_PASSWORD`** (defaults match local Postgres)

Full documentation: [CarDetector/README.md](CarDetector/README.md).
