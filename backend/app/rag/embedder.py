"""Text embedding functionality."""
from typing import List
import logging
import requests
from app.core.config import settings
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Together API endpoint (correct endpoint, same as embed_sentences.py)
TOGETHER_EMBEDDINGS_URL = "https://api.together.xyz/v1/embeddings"


class Embedder:
    """Handles text embedding using Together AI API directly."""

    def __init__(self):
        """Initialize the embedder with API configuration."""
        self.api_key = settings.TOGETHER_API_KEY
        self.model = settings.EMBED_MODEL
        self.api_url = TOGETHER_EMBEDDINGS_URL
        logger.info(f"Initialized Embedder with model: {self.model}")

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string into a vector.

        Args:
            query: Query text string

        Returns:
            Embedding vector as list of floats
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.model,
                "input": [query],
            }

            logger.debug(f"Requesting embedding for model: {self.model}")
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            # Together API returns: {"data": [{"embedding": [...], "index": 0}, ...]}
            embedding = data["data"][0]["embedding"]
            
            # Check if API returned model info (some APIs include this)
            actual_model_used = data.get("model", self.model)
            if actual_model_used != self.model:
                logger.warning(f"API used model '{actual_model_used}' instead of requested '{self.model}'")
            
            # Log full response for debugging (first time only)
            if not hasattr(self, '_logged_response'):
                logger.info(f"API Response structure: {list(data.keys())}")
                if "data" in data and len(data["data"]) > 0:
                    item_keys = list(data["data"][0].keys())
                    logger.info(f"Embedding item keys: {item_keys}")
                logger.info(f"Requested model: {self.model}, API model: {actual_model_used}")
                self._logged_response = True
            
            # Validate embedding dimension matches expected
            expected_dim = settings.EMBEDDING_DIMENSION
            actual_dim = len(embedding)
            
            logger.info(f"Embedding generated: model={actual_model_used}, dimension={actual_dim}, expected={expected_dim}")
            
            if actual_dim != expected_dim:
                # Provide helpful suggestions based on common dimensions
                suggestions = []
                if actual_dim == 1536:
                    suggestions.append("You may need to update EMBEDDING_DIMENSION in .env to 1536")
                    suggestions.append("Or change EMBED_MODEL to a model that produces 768 dimensions (e.g., BAAI/bge-base-en-v1.5)")
                elif actual_dim == 768:
                    suggestions.append("Your Neo4j index might be configured for a different dimension")
                    suggestions.append("Recreate the vector index with: CREATE VECTOR INDEX ... OPTIONS {indexConfig: {`vector.dimensions`: 768}}")
                else:
                    suggestions.append(f"Update EMBEDDING_DIMENSION in .env to {actual_dim}")
                    suggestions.append("Or change EMBED_MODEL to match your Neo4j index dimension")
                
                error_msg = (
                    f"Embedding dimension mismatch!\n"
                    f"  Expected: {expected_dim} (from EMBEDDING_DIMENSION in .env)\n"
                    f"  Got: {actual_dim} (from model: {actual_model_used})\n"
                    f"  Requested model: {self.model}\n\n"
                    f"Solutions:\n" + "\n".join(f"  - {s}" for s in suggestions) + "\n\n"
                    f"Common model dimensions:\n"
                    f"  - BAAI/bge-base-en-v1.5: 768\n"
                    f"  - togethercomputer/m2-bert-80M-8k-retrieval: 768\n"
                    f"  - text-embedding-3-small: 1536\n"
                    f"  - text-embedding-3-large: 3072"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Generated embedding with dimension {actual_dim}")
            return embedding

        except Exception as e:
            logger.exception("Failed to generate query embedding")
            raise RuntimeError(f"Failed to generate query embedding: {e}")
