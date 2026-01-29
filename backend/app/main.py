"""FastAPI application entry point."""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.rag.schemas import QueryRequest, QueryResponse, EvidenceItem
from app.rag.retriever import Retriever
from app.rag.generator import Generator
from app.db.neo4j import check_connectivity, close_driver
import os
from dotenv import load_dotenv

load_dotenv()
# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress sensitive information in logs
logging.getLogger("openai").setLevel(logging.WARNING)

app = FastAPI(
    title="PlasmaLTP RAG API",
    description="Retrieval-Augmented Generation API",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components
logger.info("Initializing RAG components")
retriever = Retriever()
generator = Generator()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down application")
    close_driver()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "PlasmaLTP RAG API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Checks:
    - Neo4j connectivity (runs RETURN 1 query)
    - Together AI auth configured (checks if API key exists, does NOT call API)
    """
    health_status = {
        "status": "healthy",
        "checks": {
            "neo4j": False,
            "together_auth": False
        }
    }
    
    # Check Neo4j connectivity
    try:
        neo4j_healthy = check_connectivity()
        health_status["checks"]["neo4j"] = neo4j_healthy
        if not neo4j_healthy:
            health_status["status"] = "degraded"
            logger.warning("Neo4j connectivity check failed")
    except Exception as e:
        logger.error(f"Neo4j health check error: {str(e)}")
        health_status["status"] = "degraded"
        health_status["checks"]["neo4j"] = False
    
    # Check Together AI auth (just verify key exists, don't call API)
    try:
        together_configured = bool(settings.TOGETHER_API_KEY and len(settings.TOGETHER_API_KEY.strip()) > 0)
        health_status["checks"]["together_auth"] = together_configured
        if not together_configured:
            health_status["status"] = "degraded"
            logger.warning("Together AI API key not configured")
    except Exception as e:
        logger.error(f"Together AI auth check error: {str(e)}")
        health_status["status"] = "degraded"
        health_status["checks"]["together_auth"] = False
    
    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main RAG query endpoint.
    
    Retrieves relevant evidence and generates a response with citations.
    """
    try:
        # How many evidence items to return (distinct papers)
        top_k = request.top_k if request.top_k is not None else settings.TOP_K_RESULTS
        # Retrieve more so we can merge by DOI and still have top_k distinct papers
        fetch_k = max(top_k * 4, 12)
        evidence = retriever.retrieve(request.question, top_k=fetch_k)
        
        # Guardrail: If no evidence returned, return insufficient evidence message
        if not evidence:
            logger.warning(f"No evidence found for query: {request.question}")
            return QueryResponse(
                answer="Insufficient evidence in corpus.",
                evidence=[]
            )
        
        # Merge by DOI: same-DOI items become one with concatenated sentences; take top_k by score
        filtered_evidence = retriever.merge_evidence_by_doi(
            evidence,
            max_items=top_k,
            sentence_sep=" ",
        )
        
        if not filtered_evidence:
            filtered_evidence = evidence[:top_k]
        
        # Generate response using retrieved evidence
        response = generator.generate(
            query=request.question,
            evidence=filtered_evidence
        )
        
        # Convert evidence to EvidenceItem objects
        evidence_items = [
            EvidenceItem(**ev) for ev in filtered_evidence
        ]
        
        return QueryResponse(
            answer=response["answer"],
            evidence=evidence_items
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )

