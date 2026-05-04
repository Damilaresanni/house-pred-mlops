# ML Elite Template

## Structure
- src/ → core code
- infra/ → infrastructure (cluster, networking)
- deploy/ → application deployment
- observability/ → monitoring + dashboards
- load_tests/ → load testing
- performance/ → benchmarking

## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run API
```
uvicorn src.api.app:app --reload
```

## Run MLflow
```
mlflow ui
```

## Run DVC pipeline
```
dvc repro
```
