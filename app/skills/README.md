# RAG Skills

Each RAG capability should live in a dedicated skill module under `app/skills/`.

## Structure

- `base.py` defines the shared skill contract.
- `registry.py` selects and executes skills.
- `factories.py` wires the default skill registry for the API.
- `analytics/` contains direct analytics-fetching skills.
- `insights/` contains derived insight skills.

## How to add a new capability

1. Create a new file in the appropriate category, for example `app/skills/insights/subscription_detection_skill.py`.
2. Implement a `Skill` subclass with:
   - `skill_id`
   - `context_key`
   - `description`
   - `keywords`
   - `execute(request)`
3. Export the class from that category package `__init__.py`.
4. Register the skill in `app/skills/factories.py`.
5. Add a focused unit test under `tests/`.

## Design rule

If a new user-facing RAG capability is added, it should be introduced as a skill file instead of expanding `app/services/rag_service.py`.

