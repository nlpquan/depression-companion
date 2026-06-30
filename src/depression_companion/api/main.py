"""FastAPI application entry point."""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from depression_companion.api.routes import router
from depression_companion.config import load_config
from depression_companion.logging_config import setup_logging

# Global startup time
START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("=" * 50)
    logger.info("Depression Companion API Starting...")
    logger.info("=" * 50)
    
    # Load configuration
    config = load_config()
    setup_logging(config.logging)
    
    # Initialize models (lazy loading for faster startup)
    logger.info("Models will be loaded on first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="Depression Companion API",
        description="Multimodal depression detection and monitoring system",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://depression-companion.vercel.app",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    return app


app = create_app()


# ---- Run with: uvicorn src.depression_companion.api.main:app --reload ----

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "depression_companion.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )