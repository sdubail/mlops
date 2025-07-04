# MLOps Cloud Run Demo

An MLOps learning project with deployment on Google Cloud Run.

## Project Setup

### 1. Install dependencies with uv

```bash
# Initialize project
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
```

### 2. Project structure

```
mlops/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── model.py             # ML model logic
│   └── schemas.py           # Pydantic schemas
├── tests/
│   └── test_api.py
├── Dockerfile
├── .dockerignore
├── pyproject.toml
├── README.md
└── .gitignore
```

## Development

### Run API locally
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

### Tests
```bash
pytest tests/
```

### Formatting and linting
```bash
# Format code
ruff format src/ tests/

# Linting
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/
```

## Deployment

### Build Docker image
```bash
docker build -t mlops-demo .
```

### Test locally with Docker
```bash
docker run -p 8080:8080 mlops-demo
```

### Deploy to Cloud Run
```bash
# Tag for Google Container Registry
docker tag mlops-demo gcr.io/YOUR-PROJECT-ID/mlops-demo

# Push to GCR
docker push gcr.io/YOUR-PROJECT-ID/mlops-demo

# Deploy to Cloud Run
gcloud run deploy mlops-demo \
  --image gcr.io/YOUR-PROJECT-ID/mlops-demo \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated
```

## API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /predict` - Make a prediction

### Usage example
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0]}'
```