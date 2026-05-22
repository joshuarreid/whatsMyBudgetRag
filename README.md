# Finance Intelligence API

This service is a thin FastAPI orchestration layer on top of Spring Boot. Spring Boot remains the system of record for transactions and analytics. FastAPI handles request shaping, lightweight aggregation, and optional LLM-based explanation.

## Architecture

- No vector database
- No embeddings
- No duplicated analytics SQL
- Spring Boot owns transaction and analytics endpoints
- FastAPI owns orchestration and natural-language answer generation

## Project Structure

```text
.
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── analytics.py
│   │       └── rag.py
│   ├── clients/
│   │   └── spring_boot_client.py
│   ├── core/
│   │   └── config.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── analytics_service.py
│   │   ├── llm_service.py
│   │   └── rag_service.py
│   ├── db_client.py
│   ├── ingest.py
│   ├── main.py
│   └── rag.py
├── Dockerfile
├── README.md
└── requirements.txt
```

The legacy modules remain only as compatibility shims while the app runs through the layered modules above.

## Configuration

Set these environment variables for local development or deployment:

```env
SPRING_BOOT_BASE_URL=http://springboot-api
HTTP_TIMEOUT_SECONDS=10
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
DEFAULT_ANALYTICS_PERIOD=2026-01
```

Notes:

- `SPRING_BOOT_BASE_URL` should point at the Spring Boot service.
- If `OPENAI_API_KEY` is unset, the `/rag/ask` endpoint still works and returns a deterministic summary from the fetched API context.
- `DEFAULT_ANALYTICS_PERIOD` is an optional fallback. If it is unset, the RAG service asks Spring Boot for available periods and uses the latest one.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

## API Endpoints

### Health

```bash
curl http://localhost:8080/health
```

### Periods

```bash
curl http://localhost:8080/analytics/periods
```

### Period Overview

```bash
curl "http://localhost:8080/analytics/periods/2026-01/overview"
```

### Category Breakdown

```bash
curl "http://localhost:8080/analytics/periods/2026-01/categories"
```

### Account Breakdown

```bash
curl "http://localhost:8080/analytics/periods/2026-01/accounts"
```

### Payment Method Breakdown

```bash
curl "http://localhost:8080/analytics/periods/2026-01/payment-methods"
```

### Daily Totals

```bash
curl "http://localhost:8080/analytics/periods/2026-01/daily"
```

### Criticality Breakdown

```bash
curl "http://localhost:8080/analytics/periods/2026-01/criticality"
```

### Duplicates

```bash
curl "http://localhost:8080/analytics/periods/2026-01/duplicates"
```

### Uncategorized

```bash
curl "http://localhost:8080/analytics/periods/2026-01/uncategorized"
```

### Outliers

```bash
curl "http://localhost:8080/analytics/periods/2026-01/outliers?limit=20"
```

### Natural-Language Ask

```bash
curl -X POST http://localhost:8080/rag/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Which accounts drove most of my spending in this period?",
    "period": "2026-01",
    "payment_method": "Credit Card",
    "transaction_id": "trace-123"
  }'
```

The RAG service does not do retrieval from a vector store. It selects relevant Spring Boot analytics endpoints based on the question, fetches the relevant period-scoped data, and optionally uses an LLM to turn that structured context into a response.

## Docker

```bash
docker build -t finance-intelligence-api .
docker run --env-file .env -p 8080:8080 finance-intelligence-api
```

## Design Constraints

- FastAPI should not write financial data.
- FastAPI should not reimplement analytics queries that belong in Spring Boot.
- FastAPI should orchestrate, normalize, and explain.
