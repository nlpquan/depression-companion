# Depression Companion - System Architecture

## Overview

The Depression Companion is a multimodal AI system for depression detection and monitoring. It analyzes speech patterns and text to provide clinical-grade insights, mood forecasting, and CBT-informed interventions.

## High-Level Architecture
┌─────────────────────────────────────────────────────────────┐
│ Client Layer │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Web App │ │ Mobile App │ │ API Client │ │
│ │ (Next.js) │ │ (Future) │ │ (3rd Party) │ │
│ └──────┬───────┘ └──────┬───────┘ └──────┬────────┘ │
└─────────┼──────────────────┼──────────────────┼──────────────┘
│ │ │
└──────────────────┼──────────────────┘
│
┌────────▼────────┐
│ API Gateway │
│ (FastAPI) │
└────────┬────────┘
│
┌──────────────────┼──────────────────┐
│ │ │
┌─────────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│ Model Serving │ │ RAG System │ │ Analytics │
│ ┌────────────┐ │ │ ┌────────┐ │ │ ┌────────────┐ │
│ │ Wav2Vec2 │ │ │ │ChromaDB│ │ │ │ Forecasting│ │
│ │ BERT │ │ │ │Retrieve│ │ │ │ Early Warn │ │
│ │ Mistral-7B │ │ │ │Generate│ │ │ │ Reports │ │
│ └────────────┘ │ │ └────────┘ │ │ └────────────┘ │
└──────────────────┘ └─────────────┘ └──────────────────┘
│
┌──────────────────┼──────────────────┐
│ │ │
┌─────────▼────────┐ ┌──────▼──────┐ ┌────────▼────────┐
│ PostgreSQL │ │ Redis │ │ Object Store │
│ (User Data) │ │ (Cache) │ │ (Audio Files) │
└──────────────────┘ └─────────────┘ └──────────────────┘

## Component Details

### API Layer (FastAPI)
- **Framework**: FastAPI (async Python)
- **Port**: 8000
- **Endpoints**: 
  - POST /api/v1/analyze/text
  - POST /api/v1/analyze/audio
  - POST /api/v1/analyze/multimodal
  - POST /api/v1/chat (RAG)
  - GET /api/v1/forecast/{user_id}
  - WS /api/v1/ws/{user_id} (real-time)

### Model Serving
- **Audio**: Wav2Vec2 fine-tuned on DAIC-WOZ
- **Text**: BERT fine-tuned on Reddit mental health data
- **LLM**: Mistral-7B with QLoRA for therapy dialogues
- **Fusion**: Cross-modal attention
- **Serving**: NVIDIA Triton (production), PyTorch (development)

### RAG System
- **Knowledge Base**: Structured CBT protocols
- **Vector Store**: ChromaDB
- **Embeddings**: all-MiniLM-L6-v2
- **Safety**: Two-stage crisis classifier

### Data Storage
- **PostgreSQL**: User data, mood logs, conversation history
- **Redis**: Session cache, rate limiting, real-time state
- **S3/GCS**: Audio file storage

## Scaling Strategy

### For 10M Users:
- **Throughput**: 10M users × 2 daily check-ins = 20M requests/day
- **Peak**: ~500 requests/second
- **Model Inference**: 30× A100 GPUs across 4 regions
- **API Servers**: 50+ FastAPI instances behind load balancer
- **Database**: PostgreSQL with read replicas, connection pooling
- **Caching**: Multi-layer (Browser → CDN → Redis → Database)
- **Estimated Cost**: ~$500K/month at scale

## Security & Privacy
- HIPAA-compliant data handling
- End-to-end encryption
- Data anonymization
- 30-day data retention policy
- Regular security audits