"""
Solvix AI Engine - FastAPI Application

Main entry point for the AI Engine service providing:
- Email classification
- Draft generation  
- Gate evaluation
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.config.settings import settings
from src.api.routes import classify, generate, gates, health

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*60)
    logger.info("Starting Solvix AI Engine")
    logger.info("="*60)
    logger.info(f"Model: {settings.openai_model}")
    logger.info(f"Port: {settings.api_port}")
    logger.info(f"Debug: {settings.debug}")
    yield

# Create app
app = FastAPI(
    title="Solvix AI Engine",
    description="AI-powered email classification and draft generation for debt collection",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(classify.router, tags=["Classification"])
app.include_router(generate.router, tags=["Generation"])
app.include_router(gates.router, tags=["Gates"])




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
