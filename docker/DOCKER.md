# Docker Setup

This project is containerized with:
- `Dockerfile`: base image for the flight fare project.
- `Dockerfile.airflow`: image for Airflow services with ML pipeline dependencies.
- `docker-compose.yml`: runs API + Streamlit together.

## Prerequisites
- Docker Desktop (or Docker Engine + Compose plugin)

## Build and Run
From the project root:

```powershell
docker compose up --build
```

## Access
- API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`
- Airflow UI: `http://localhost:8080` (user: `admin`, password: `admin`)

## Stop

```powershell
docker compose down
```

## Notes
- The API expects packaged artifacts at:
  - `/app/artifacts/stage_7/package/v1`
- Host folders `./artifacts` and `./logs` are mounted into both containers.
- Airflow services use:
  - DAGs: `/opt/airflow/dags` (mapped from `./dags`)
  - Project source: `/opt/airflow/src` (mapped from `./src`)
  - Dataset: `/opt/airflow/notebook/raw_bangladesh_data/Flight_Price_Dataset_of_Bangladesh.csv`
  - Airflow DB: Postgres service on compose network (`airflow-db`)

## Airflow DAG
After services are up, open Airflow UI and trigger:
- `flight_fare_stages_2_to_7`
