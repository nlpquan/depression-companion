# ADR-004: Vercel + Railway over AWS for MVP

## Status
Accepted (MVP) / Proposed (Production)

## Context
Need to deploy the Depression Companion MVP quickly with zero cost, while designing for FAANG-scale production.

## Decision

### MVP (Current): Vercel (Frontend) + Railway (Backend)
- Free tier sufficient for portfolio demo
- Automatic deployments from GitHub
- Zero DevOps overhead
- Limited to <100 concurrent users

### Production (Designed): AWS/GCP with Kubernetes
- EKS/GKE for container orchestration
- NVIDIA Triton for model serving (3x throughput via dynamic batching)
- Horizontal Pod Autoscaling based on inference queue depth
- Blue-green deployment with <1% error rate rollback
- Multi-region deployment (us-east, eu-west, ap-southeast)

## Migration Path
1. Containerize with Docker (done)
2. Deploy to Kubernetes (Helm charts ready)
3. Add monitoring (Prometheus + Grafana)
4. Implement canary deployments
5. Multi-region failover

## Cost Comparison
| Stage | Service | Monthly Cost |
|-------|---------|-------------|
| MVP | Vercel + Railway | $0-20 |
| Beta | Single AWS instance | $200-500 |
| Production | K8s cluster + GPUs | $500K |