# ADR-003: Redis over Memcached for Caching

## Status
Accepted

## Context
Need a caching layer for session state, real-time analysis results, rate limiting, and model inference cache.

## Decision
Use **Redis** instead of Memcached.

## Rationale

### Advantages of Redis:
1. **Rich data structures** - Lists, sets, sorted sets for mood history, leaderboards
2. **Pub/Sub** - Enables real-time WebSocket broadcasting
3. **Persistence** - RDB/AOF for cache durability
4. **Lua scripting** - Atomic operations for rate limiting
5. **Stream data type** - Perfect for time-series mood data

### Why not Memcached:
1. Limited to key-value only
2. No persistence
3. No pub/sub for real-time features

## Cache Strategy
- L1: Browser cache (mood history, static assets)
- L2: CDN (model weights, frontend assets)  
- L3: Redis (API responses, session state, inference cache)
- L4: PostgreSQL (persistent data)

## Estimated Cache Hit Rate: 60-70%