# Conversation History Architecture

This document explains how a frontend should communicate with the FastAPI backend to implement a conversational finance chatbot.

The short version:

- the frontend talks only to the API
- the API owns conversation creation and continuation
- the API persists chat history in MySQL
- the frontend stores and reuses `conversation_id`
- the frontend never talks directly to MySQL

## Goals

The current backend supports a conversational RAG flow for a **single-user** app.

That means:

- every saved conversation is stored under the fixed backend owner `default-user`
- the frontend does **not** need to send authentication or user identity for conversation ownership
- the frontend only needs to manage chat state and conversation IDs

## System Boundaries

### Frontend responsibilities

The frontend should:

- render the chat UI
- send user messages to `POST /rag/ask`
- store the returned `conversation_id`
- send the same `conversation_id` on follow-up messages
- optionally fetch prior messages with `GET /rag/conversations/{conversation_id}`
- display loading, retry, and error states

### Backend responsibilities

The backend should:

- create a conversation when a message arrives without a `conversation_id`
- load prior history when a message arrives with a `conversation_id`
- build RAG context from the current question, filters, and recent history
- call Spring Boot analytics APIs
- optionally call the LLM
- persist both the user message and assistant response
- return the answer plus the active `conversation_id`

### Database responsibilities

MySQL stores:

- conversation threads in `conversations`
- user and assistant turns in `messages`
- optional tool-call snapshots in `message_tool_calls`
- optional cached tool results in `conversation_tool_cache`

The frontend should never query these tables directly.

## Recommended Frontend Flow

### 1. Start a new chat

When the user opens a new conversation:

- initialize local UI state with no `conversation_id`
- render an empty transcript
- wait for the user to send a first message

### 2. Send the first message

Send:

```http
POST /rag/ask
Content-Type: application/json
```

Request body:

```json
{
  "question": "How much did I spend this month?"
}
```

Backend behavior:

- creates a new conversation
- saves the user message
- computes the answer
- saves the assistant message
- returns a new `conversation_id`

Frontend behavior:

- append the user message locally immediately
- show an assistant loading state
- replace loading with the assistant response
- store `conversation_id` in component/app state

### 3. Continue the conversation

Every follow-up message should include the prior `conversation_id`.

Example:

```json
{
  "conversation_id": "4e9cf933-5b92-4f85-9c58-5c9a3fbe4d35",
  "question": "What about just credit card purchases?",
  "payment_method": "Credit Card"
}
```

Backend behavior:

- loads recent conversation history
- injects history into the answer context
- saves the new user and assistant turns
- returns the same `conversation_id`

### 4. Re-open an existing chat

If the user navigates away and comes back later:

```http
GET /rag/conversations/{conversation_id}?limit=50
```

Use the response to rebuild the visible transcript.

## API Contract

## `POST /rag/ask`

Primary endpoint for sending a message.

### Request fields

| Field | Type | Required | Notes |
|---|---|---:|---|
| `question` | string | yes | The user message |
| `conversation_id` | string | no | Omit for a new chat; include for follow-ups |
| `period` | string | no | Explicit statement period like `May2026` |
| `payment_method` | string | no | Optional analytics filter |
| `account` | string | no | Optional analytics filter |
| `transaction_id` | string | no | Optional trace/correlation ID |

### Example request

```json
{
  "conversation_id": "4e9cf933-5b92-4f85-9c58-5c9a3fbe4d35",
  "question": "What were my top categories?",
  "period": "May2026"
}
```

### Example response

```json
{
  "question": "What were my top categories?",
  "conversation_id": "4e9cf933-5b92-4f85-9c58-5c9a3fbe4d35",
  "period": "May2026",
  "plan": ["categories", "top_categories"],
  "tool_selection": {
    "llm_suggested_tools": [],
    "deterministic_tools": ["categories", "top_categories"],
    "union_tools": ["categories", "top_categories"]
  },
  "context": {
    "conversation": {
      "conversation_id": "4e9cf933-5b92-4f85-9c58-5c9a3fbe4d35",
      "history_message_count": 4
    },
    "conversation_history": [
      {
        "message_id": "...",
        "role": "user",
        "content": "How much did I spend this month?",
        "period": "May2026",
        "period_source": "question_current_month",
        "created_at": "2026-05-23T12:00:00+00:00"
      }
    ]
  },
  "answer": "Your top categories this month were groceries and dining."
}
```

### Frontend notes

The frontend usually needs only:

- `conversation_id`
- `answer`
- optionally `period`
- optionally `context` for debug/advanced UX

For a normal chat UI, do **not** render the full `context` by default.

## `GET /rag/conversations/{conversation_id}`

Fetches saved conversation history.

### Query params

| Param | Type | Required | Notes |
|---|---|---:|---|
| `limit` | integer | no | Default `50`, max `200` |

### Example response

```json
{
  "conversation_id": "4e9cf933-5b92-4f85-9c58-5c9a3fbe4d35",
  "title": "How much did I spend this month?",
  "created_at": "2026-05-23T12:00:00+00:00",
  "updated_at": "2026-05-23T12:05:00+00:00",
  "last_message_at": "2026-05-23T12:05:00+00:00",
  "messages": [
    {
      "message_id": "msg-1",
      "role": "user",
      "content": "How much did I spend this month?",
      "period": "May2026",
      "period_source": "question_current_month",
      "created_at": "2026-05-23T12:00:00+00:00"
    },
    {
      "message_id": "msg-2",
      "role": "assistant",
      "content": "You spent $2,150 this month.",
      "period": "May2026",
      "period_source": "question_current_month",
      "created_at": "2026-05-23T12:00:01+00:00"
    }
  ]
}
```

## Suggested Frontend State Model

A simple client-side state shape could look like this:

```ts
interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt?: string;
}

interface ChatSession {
  conversationId: string | null;
  messages: ChatMessage[];
  isSending: boolean;
  error: string | null;
}
```

## Recommended Message Send Algorithm

```text
1. User submits text
2. Frontend appends optimistic user message to UI
3. Frontend sends POST /rag/ask with current conversation_id if present
4. Backend returns answer + conversation_id
5. Frontend stores conversation_id if it was previously null
6. Frontend appends assistant message to UI
7. On reload, frontend calls GET /rag/conversations/{conversation_id}
```

## Headers and Tracing

The backend supports:

- `X-Request-ID`
- `X-Transaction-ID`

The frontend may send them if desired.

If omitted:

- the backend generates them automatically
- the backend includes them in the response headers

This is useful if you want to correlate frontend errors with backend logs.

## Error Handling Expectations

### `404 Not Found`

Typical case:

- frontend sends a `conversation_id` that does not exist

Frontend behavior:

- show a friendly message like: `This conversation could not be found.`
- offer to start a new chat

### `503 Service Unavailable`

Typical cases:

- conversation history is not configured
- database connectivity is unavailable

Frontend behavior:

- show a temporary system error
- allow retry
- optionally degrade to a stateless chat experience if desired

### `500 Internal Server Error`

Typical cases:

- unexpected backend exception
- upstream analytics failure not gracefully handled

Frontend behavior:

- keep the user’s draft or optimistic message visible
- show retry action
- avoid losing `conversation_id`

## UX Recommendations

### Keep `conversation_id` client-side

Store it in:

- React state
- route state
- URL param
- local storage

Recommended:

- keep the active `conversation_id` in component/app state
- optionally mirror it into the URL for refresh/share/reload behavior

### Render assistant responses as plain chat bubbles

The backend response contains rich structured context, but a normal user should mostly see:

- the assistant `answer`
- optionally the resolved `period`

### Rehydrate on page load

If the current page knows a `conversation_id`, call:

```http
GET /rag/conversations/{conversation_id}
```

before enabling new sends.

### Do not invent local thread IDs

Let the backend be the source of truth for `conversation_id`.

## Architectural Sequence

```text
Frontend
  -> POST /rag/ask
Backend route
  -> RAGService.answer()
RAGService
  -> create or load conversation from MySQL
  -> save user message
  -> resolve period and tool plan
  -> fetch analytics from Spring Boot
  -> optionally call LLM
  -> save assistant message
  -> return answer + conversation_id
Frontend
  -> stores conversation_id
  -> renders assistant message
```

## What the Frontend Should Not Do

The frontend should not:

- connect directly to MySQL
- manage message sequence numbers
- decide whether a conversation exists in the database
- rely on the internal MySQL schema
- build its own finance answer from raw context when the backend already returned `answer`

## Minimal Frontend Integration Example

### First message

```js
const response = await fetch("/rag/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    question: userInput,
  }),
});

const data = await response.json();
setConversationId(data.conversation_id);
appendAssistantMessage(data.answer);
```

### Follow-up message

```js
const response = await fetch("/rag/ask", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    conversation_id: currentConversationId,
    question: userInput,
  }),
});
```

### Restore transcript

```js
const response = await fetch(`/rag/conversations/${conversationId}?limit=50`);
const data = await response.json();
setMessages(data.messages);
```

## Recommended Phase Order for Frontend Work

1. Implement send message with `POST /rag/ask`
2. Store returned `conversation_id`
3. Implement follow-up sends with `conversation_id`
4. Add transcript restore with `GET /rag/conversations/{conversation_id}`
5. Add retry/error states
6. Optionally surface period/filter metadata in the UI

## Summary

For the frontend, the architecture is intentionally simple:

- start without a `conversation_id`
- send the first message to `POST /rag/ask`
- save the returned `conversation_id`
- reuse it for every follow-up
- call `GET /rag/conversations/{conversation_id}` to restore history
- let the backend own persistence, history assembly, and RAG orchestration

