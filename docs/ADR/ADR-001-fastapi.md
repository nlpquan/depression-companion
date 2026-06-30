# ADR-001: FastAPI over Flask for API Framework

## Status
Accepted

## Context
Need to choose a Python web framework for the Depression Companion API. The API will handle synchronous (REST) and asynchronous (WebSocket) requests, serve ML models, and require auto-generated documentation.

## Decision
Use **FastAPI** instead of Flask.

## Rationale

### Advantages of FastAPI:
1. **Native async support** - Critical for WebSocket connections and concurrent model inference
2. **Automatic OpenAPI docs** - Swagger UI and ReDoc generated automatically
3. **Pydantic integration** - Same validation library used in our config system
4. **Performance** - On par with Node.js/Go due to Starlette/uvicorn
5. **Type safety** - Full type hint support matching our codebase standards
6. **Dependency injection** - Clean architecture for shared resources (models, DB)

### Why not Flask:
1. Limited async support (requires extensions)
2. Manual API documentation
3. Less performant for real-time applications

## Consequences
- All API routes use async/await
- Request/response validation is automatic via Pydantic
- Interactive API docs available at /docs