# Services Layer Notes

This folder contains the orchestration layer for the budgeting RAG application. The service classes sit between:

- FastAPI routes in `app/api/routes/`
- typed request/response models in `app/models/schemas.py`
- deterministic skill execution in `app/skills/`
- optional LLM-assisted routing and answer generation

Two important building blocks in this project are **Pydantic** and **LangGraph**.

---

## How Pydantic works in this project

Pydantic is used primarily for **typed contracts** and **validation**.

The main schema definitions live in `app/models/schemas.py`, and the services in this folder consume those models rather than passing around loose dictionaries wherever possible.

### 1. API request and response validation

The RAG route in `app/api/routes/rag.py` uses Pydantic models as FastAPI request and response types:

- `RagAskRequest` validates incoming `POST /rag/ask` payloads
- `RagAnswerResponse` defines the shape of the final response
- `RagConversationResponse` defines the conversation-history response

That means FastAPI automatically:

- parses incoming JSON into Python objects
- validates required fields and types
- serializes returned models back to JSON

Example from the route layer:

- request body -> `RagAskRequest`
- service result -> `RagAnswerResponse`

### 2. Shared service-layer types

Several services depend on shared schema models so they can coordinate without duplicating validation logic.

Important examples:

- `RagTimeScope`
  - used throughout the service layer to represent a single period, a period range, or a date range
  - consumed by `RAGService`, `PlannerService`, and `LangGraphReasoningService`
- `RagExecutionPlan` and `RagPlanStep`
  - returned by `PlannerService` to describe which skills should run and with what scope
- `RagIntentResponse`
  - used by `IntentService` for LLM-based intent classification output
- `RagToolTraceResponse`, `RagCitationResponse`, `RagTimingMetadataResponse`, `RagCacheMetadataResponse`
  - used to build the final structured answer payload

### 3. Validation rules are centralized in the models

`RagTimeScope` is the best example of why Pydantic is useful here.

It does more than hold data. It enforces the allowed combinations of fields:

- `scope_type="statement_period"` requires `statement_period`
- `scope_type="statement_period_range"` requires `start_period` and `end_period`
- `scope_type="date_range"` requires `start_date` and `end_date`
- incompatible field combinations are rejected
- statement periods must use the `MonthYYYY` format, such as `May2026`

This validation happens through:

- `@field_validator(...)`
- `@model_validator(mode="after")`

That keeps time-scope validation out of the individual services and gives the whole app one consistent rule set.

### 4. Aliases make external payloads easier to consume

Some analytics and insight responses use Java/Spring-style field names such as:

- `totalAmount`
- `transactionCount`
- `paymentMethod`
- `statementPeriod`

Pydantic models map those into Python-friendly attributes using `Field(alias=...)`.

Example:

- JSON field `totalAmount`
- Python attribute `total_amount`

Because `AnalyticsBaseModel` uses `ConfigDict(populate_by_name=True)`, the code can work with Pythonic names while still accepting aliased API payloads.

### 5. Services convert untyped data into typed models

You will see a few common Pydantic conversion patterns in the services:

- `model_validate(...)`
  - convert raw dict-like data into a Pydantic model
- `model_dump(...)`
  - convert a model back into JSON-safe output for logging, persistence, or API responses

Examples already used in the service layer:

- `IntentService` asks the OpenAI Responses API to parse structured output directly into `RagIntentResponse`
- `RAGService` returns `RagAnswerResponse`
- `RAGService` logs and persists structured data using `.model_dump(mode="json", exclude_none=True)`
- citations are normalized with `RagCitationResponse.model_validate(...)`

### 6. Pydantic is used for structure, not everything

Not every object in the service layer is a Pydantic model.

Examples of non-Pydantic structures in this project:

- `Settings` in `app/core/config.py` is a frozen dataclass
- `IntentPattern` in `app/services/langgraph_reasoning_service.py` is a frozen dataclass
- LangGraph internal state uses a `TypedDict`

This is intentional:

- use **Pydantic** when validation/serialization matters
- use **dataclasses** or **TypedDict** when lightweight internal structures are enough

---

## How LangGraph works in this project

LangGraph is used in a very focused way: it helps the app **expand skill selection** for more complex finance questions.

The main implementation is in `app/services/langgraph_reasoning_service.py`.

### 1. LangGraph is optional

The project does not hard-require LangGraph at runtime.

`LangGraphReasoningService` imports LangGraph inside a `try/except` block:

- if `langgraph` is installed, the graph is enabled
- if not, the service logs a warning and falls back to legacy routing behavior

The feature is also gated by configuration:

- `LANGGRAPH_ENABLED` environment variable
- loaded through `get_settings()` in `app/core/config.py`
- wired into the route/service setup in `app/api/routes/rag.py`

In `get_rag_service()`:

- `LangGraphReasoningService(enabled=settings.langgraph_enabled)` is constructed
- then injected into `RAGService`

### 2. What problem LangGraph solves here

The app already has deterministic skill matching and optional LLM intent classification.

LangGraph adds a middle layer for questions that need **reasoning over tool families**, for example:

- comparisons across months
- diagnostic questions like “why did my spend jump?”
- anomaly/pattern questions like “show outliers and daily trends”
- broad follow-up questions like “what changed?”

Instead of only picking one directly matched skill, the graph can expand the request into a set of related skills such as:

- baseline summaries
- driver analysis tools
- pattern/anomaly tools
- narrative insight tools

### 3. Graph state in this project

The graph state is defined as `_ReasoningState`, a `TypedDict`.

It carries information such as:

- `question`
- `time_scope`
- `available_skill_ids`
- `seed_skill_ids`
- `selected_skill_ids`
- `selected_families`
- `intents`
- `graph_trace`

A few important details:

- `time_scope` is a `RagTimeScope` Pydantic model
- `intents` is a plain `dict[str, bool]`
- `graph_trace` records a readable audit trail of graph decisions

### 4. The node flow is intentionally simple

The graph is built in `_build_graph()` with four nodes:

1. `assess_question`
2. `select_baseline`
3. `expand_reasoning_families`
4. `finalize_selection`

Flow:

`START -> assess_question -> select_baseline -> expand_reasoning_families -> finalize_selection -> END`

This is not a branching agent workflow. It is a lightweight deterministic state machine.

### 5. What each node does

#### `assess_question`

This node turns the natural-language question into intent flags.

It now uses structured `IntentPattern` definitions rather than brittle flat keyword tuples.
Those patterns support:

- strong multi-word phrases
- supporting phrases
- token groups
- regex patterns
- overlap between intents

Examples of detected intents include:

- `comparison`
- `diagnostic`
- `trend`
- `anomaly`
- `summary`
- `average`
- `available_periods`
- `category_focus`
- `account_focus`
- `payment_focus`
- `criticality_focus`
- `duplicates_focus`
- `uncategorized_focus`
- `broad_follow_up`

#### `select_baseline`

This node chooses the foundational summary skill(s) to provide context for the answer.

Examples:

- use `statement_period_summary_range` for range/comparison questions when available
- use `statement_period_summary` and/or `overview` for diagnostic/summary follow-ups
- use `available_periods` for history/coverage questions

If baseline tools are added, the graph records the `baseline_summary` family.

#### `expand_reasoning_families`

This node adds related skill families based on the detected intents.

Families currently used:

- `baseline_summary`
- `driver_analysis`
- `pattern_anomaly`
- `derived_narrative`

Examples:

- comparison/diagnostic/account/category/payment questions can add driver tools
- trend/anomaly/duplicates/uncategorized questions can add pattern tools
- comparison/summary/average questions can add insight/narrative tools

This is where the graph turns one user question into a richer multi-tool plan.

#### `finalize_selection`

This node removes anything that is not actually available in the current registry.

That keeps the graph safe even if the intent logic asks for a skill that is not installed or not registered in the current runtime.

### 6. How LangGraph fits into the full request flow

The high-level path looks like this:

1. FastAPI receives `RagAskRequest`
2. `RAGService.answer(...)` resolves time scope, account filters, and routing policy
3. `RAGService._select_skills(...)` starts from deterministic/LLM-selected skills
4. if LangGraph is enabled, `LangGraphReasoningService.plan(...)` can expand the skill set
5. `PlannerService.build_plan(...)` converts selected skills into a `RagExecutionPlan`
6. the plan is executed and results are assembled into `RagAnswerResponse`

So LangGraph does **not** execute tools itself.
It only improves **which skills should be selected together**.

### 7. Why the graph is useful here

The graph adds two benefits:

- **better explainability**
  - `graph_trace` shows which intents and skill families were chosen
- **better tool composition**
  - a question like “Why did my spend jump in April versus March?” can trigger:
    - a baseline comparison summary
    - driver tools such as categories/accounts/payment methods
    - narrative tools such as month-over-month analysis
    - anomaly tools such as outliers

That is hard to express cleanly with only one-off keyword checks.

---

## How Pydantic and LangGraph work together

These two pieces serve different roles:

- **Pydantic** = safe data contracts
- **LangGraph** = routing and reasoning over tool selection

In practice:

- Pydantic makes sure `RagTimeScope`, intent responses, plans, traces, and API payloads are valid
- LangGraph uses those validated values to make higher-level skill-selection decisions

A concrete example:

1. A request arrives with a `time_scope`
2. FastAPI/Pydantic parse it into `RagTimeScope`
3. `RAGService` passes that typed scope into `LangGraphReasoningService.plan(...)`
4. the graph uses the scope plus detected intents to choose skill families
5. `PlannerService` returns a typed `RagExecutionPlan`
6. `RAGService` returns a typed `RagAnswerResponse`

The graph provides orchestration, while Pydantic provides safety and consistency around the data moving through that orchestration.

---

## Maintenance notes

If you extend this folder, these rules will keep the design consistent:

### When to add or update a Pydantic model

Use `app/models/schemas.py` when you need:

- request/response validation
- consistent JSON serialization
- field aliases
- reusable typed payloads shared across services
- validators for business rules like time-scope consistency

### When to update LangGraph logic

Update `app/services/langgraph_reasoning_service.py` when you need to change:

- how questions are interpreted into intents
- which skill families are expanded for certain intents
- how graph metadata is traced for debugging

Try to keep these concerns separate:

- **intent detection** in `assess_question`
- **baseline selection** in `select_baseline`
- **family expansion** in `expand_reasoning_families`
- **availability filtering** in `finalize_selection`

### When not to use LangGraph

LangGraph is not the right place for:

- parsing request JSON
- validating API payloads
- executing external HTTP calls
- formatting final API responses

Those concerns already belong elsewhere:

- Pydantic / FastAPI for validation and serialization
- service classes and skills for execution
- `RAGService` for orchestration

---

## Quick reference

### Pydantic files

- `app/models/schemas.py`
- `app/api/routes/rag.py`
- `app/services/rag_service.py`
- `app/services/intent_service.py`
- `app/services/planner_service.py`

### LangGraph files

- `app/services/langgraph_reasoning_service.py`
- `app/api/routes/rag.py`
- `app/core/config.py`
- `app/services/rag_service.py`

### Important symbols

- `RagTimeScope`
- `RagAskRequest`
- `RagAnswerResponse`
- `RagIntentResponse`
- `RagExecutionPlan`
- `LangGraphReasoningService`
- `_ReasoningState`
- `IntentPattern`

---

## Summary

In this project:

- Pydantic gives the services a reliable typed contract for requests, plans, and responses
- LangGraph gives the services a lightweight reasoning layer for choosing related skill families
- `RAGService` is the place where those two ideas come together during request handling

If you are new to this codebase, start with these files in order:

1. `app/models/schemas.py`
2. `app/api/routes/rag.py`
3. `app/services/rag_service.py`
4. `app/services/intent_service.py`
5. `app/services/langgraph_reasoning_service.py`
6. `app/services/planner_service.py`

