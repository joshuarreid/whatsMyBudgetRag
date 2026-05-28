# Finance Intelligence API

This service is a thin FastAPI orchestration layer on top of Spring Boot. Spring Boot remains the system of record for transactions and analytics. FastAPI handles request shaping, lightweight aggregation, and optional LLM-based explanation.

## Architecture

- No vector database
- No embeddings
- No duplicated analytics SQL
- Spring Boot owns transaction and analytics endpoints
- FastAPI owns orchestration and natural-language answer generation
- Derived intelligence is built from structured analytics, not embeddings

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
│   │   ├── insight_service.py
│   │   ├── llm_service.py
│   │   ├── normalizers.py
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
LOG_LEVEL=INFO
LOG_FORMAT=text
LANGGRAPH_ENABLED=false
CORS_ENABLED=false
CORS_ALLOWED_ORIGINS=
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4o-mini
MYSQL_HOST=
MYSQL_PORT=25060
MYSQL_DATABASE=budget_rag
MYSQL_USER=doadmin
MYSQL_PASSWORD=
MYSQL_SSL_DISABLED=false
MYSQL_SSL_CA=
MYSQL_CONNECT_TIMEOUT_SECONDS=10
CONVERSATION_DEFAULT_USER=default-user
CONVERSATION_HISTORY_CONTEXT_LIMIT=10
INSIGHT_HIGH_SHARE_THRESHOLD=45
INSIGHT_OUTLIER_AMOUNT_THRESHOLD=500
```

Notes:

- `SPRING_BOOT_BASE_URL` should point at the Spring Boot service.
- `LOG_LEVEL` controls application and request logging verbosity.
- `LOG_FORMAT` accepts `text` or `json` for human-readable or structured logs.
- `LANGGRAPH_ENABLED` turns on graph-based skill-family reasoning for multi-step finance questions such as comparisons, driver analysis, anomaly investigation, and narrative summaries.
- `CORS_ENABLED` toggles FastAPI CORS middleware on or off.
- `CORS_ALLOWED_ORIGINS` accepts a comma-separated list of allowed browser origins when CORS is enabled.
- If `OPENAI_API_KEY` is unset, the `/rag/ask` endpoint still works and returns a deterministic summary from the fetched API context.
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_DATABASE`, `MYSQL_USER`, and `MYSQL_PASSWORD` configure conversation history persistence in MySQL 8, including managed providers such as DigitalOcean.
- `MYSQL_SSL_DISABLED=false` keeps TLS enabled for managed MySQL services; optionally set `MYSQL_SSL_CA` when your provider gives you a CA bundle path.
- `CONVERSATION_DEFAULT_USER` defaults every saved conversation to a fixed owner such as `default-user` for this single-user app.
- `CONVERSATION_HISTORY_CONTEXT_LIMIT` controls how many prior messages are injected into follow-up RAG prompts.
- Questions without an explicit or inferred time reference now fall back to the current statement period, formatted as `MonthYear` (for example `May2026`).
- `INSIGHT_HIGH_SHARE_THRESHOLD` controls when concentration warnings are emitted.
- `INSIGHT_OUTLIER_AMOUNT_THRESHOLD` controls the amount threshold for derived outlier flags.

For local frontend development against a deployed API, a typical `.env` example is:

```env
CORS_ENABLED=true
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

The natural-language endpoint remains `POST /rag/ask`, so browser clients should target `/rag/ask` rather than `/ask`.

## LangGraph Reasoning Mode

When `LANGGRAPH_ENABLED=true`, the RAG router can expand beyond direct keyword matches and select additional skill families using a LangGraph state machine:

- **Baseline / summary**: `overview`, `available_periods`, `statement_period_summary`, `statement_period_summary_range`
- **Driver analysis**: `categories`, `top_categories`, `account_breakdown`, `payment_methods`, `criticality`
- **Pattern / anomaly**: `daily`, `duplicates`, `outliers`, `uncategorized`
- **Derived narrative**: `averages`, `month_over_month`, `period_summary`, `behavior_summary`

This is most useful for questions such as:

- “Why did my spending jump in April versus March?”
- “Show my daily trend and highlight outliers.”
- “What should I focus on this month?”
- “Compare two months and tell me what drove the change.”

The response context will include `routing.reasoning_graph` metadata so you can inspect which families and skills the graph selected.

Outbound Spring Boot requests now automatically include `X-Transaction-ID` and `X-Request-ID`. If an inbound request provides `X-Transaction-ID`, that value is reused; otherwise the app falls back to the current request ID.

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8080
LOG_FORMAT=json .venv/bin/python -m uvicorn app.main:app --reload --port 8080
```

## Startup Command

From the project root, use this command:

```bash
.venv/bin/python -m uvicorn app.main:app --reload --port 8080
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
    "conversation_id": "existing-conversation-id-optional",
    "period": "May2026",
    "payment_method": "Credit Card",
    "transaction_id": "trace-123"
  }'
```

The RAG service does not do retrieval from a vector store. It selects relevant Spring Boot analytics endpoints based on the question, fetches the relevant period-scoped data, and optionally uses an LLM to turn that structured context into a response.
When MySQL conversation history is configured, `/rag/ask` will create a conversation automatically when `conversation_id` is omitted and return the saved `conversation_id` in the response.

### Conversation History

```bash
curl "http://localhost:8080/rag/conversations/<conversation_id>?limit=50"
```

### Phase 2 Summary

```bash
curl "http://localhost:8080/insights/periods/2026-01/summary"
```

### Phase 2 Behavior

```bash
curl "http://localhost:8080/insights/periods/2026-01/behavior"
```

### Phase 2 Averages

```bash
curl "http://localhost:8080/insights/periods/2026-01/averages"
```

### Phase 2 Month Over Month

```bash
curl "http://localhost:8080/insights/periods/2026-01/month-over-month"
```

These endpoints build a derived intelligence layer on top of the existing analytics contract. They add concentration metrics, anomaly flags, spend-share summaries, and period behavior lines without introducing embeddings or a vector database.
They now also expose period averages and month-over-month comparison metrics using the previous available statement period.

## Docker

```bash
docker build -t finance-intelligence-api .
docker run --env-file .env -p 8080:8080 finance-intelligence-api
```

## Design Constraints

- FastAPI should not write financial data.
- FastAPI should not reimplement analytics queries that belong in Spring Boot.
- FastAPI should orchestrate, normalize, and explain.
