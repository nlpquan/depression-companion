# ADR-002: PostgreSQL over MongoDB

## Status
Accepted

## Context
Need a database for user data, mood logs, conversation history, and clinical reports.

## Decision
Use **PostgreSQL** instead of MongoDB.

## Rationale

### Advantages of PostgreSQL:
1. **ACID compliance** - Critical for clinical data integrity
2. **Relational data** - Users have clear relationships (user → mood_logs, user → conversations)
3. **JSON support** - JSONB column type provides NoSQL flexibility when needed
4. **Mature ecosystem** - pgvector for embeddings, connection pooling, monitoring
5. **HIPAA compliance** - Strong encryption, audit logging, row-level security

### Why not MongoDB:
1. Eventual consistency is unacceptable for clinical data
2. Complex transactions are harder to implement
3. Schema flexibility can lead to data inconsistency

## Consequences
- Use SQLAlchemy + Alembic for ORM and migrations
- JSONB columns for flexible feature storage
- Read replicas for scaling