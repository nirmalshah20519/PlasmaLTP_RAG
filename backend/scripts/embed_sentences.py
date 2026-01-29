"""
Script to embed sentences and store embeddings in Neo4j
(using Together AI API with correct endpoint).
"""

import sys
import time
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import requests
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.db.neo4j import execute_query, execute_write, close_driver

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Together API Configuration
# ---------------------------------------------------------------------
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')
TOGETHER_API_URL = "https://api.together.xyz/v1/embeddings"  # Correct endpoint
BATCH_SIZE = os.environ.get('BATCH_SIZE')
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY environment variable is not set")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_text(text: str, max_chars: int = 1500) -> str:
    """
    Trim text to avoid embedding failures and latency spikes.
    """
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def count_sentences_to_embed(force_reembed: bool = False) -> int:
    """
    Return total number of Sentence nodes to process (missing embeddings or all with text).
    """
    if force_reembed:
        query = """
        MATCH (s:Sentence)
        WHERE s.text IS NOT NULL AND trim(s.text) <> ""
        RETURN count(s) AS n
        """
    else:
        query = """
        MATCH (s:Sentence)
        WHERE s.text IS NOT NULL
          AND trim(s.text) <> ""
          AND (s.embedding IS NULL OR size(s.embedding) = 0)
        RETURN count(s) AS n
        """
    result = execute_query(query)
    return result[0]["n"] if result else 0


def fetch_sentences_missing_embeddings(
    limit: int, force_reembed: bool = False, skip: int = 0
) -> List[Dict[str, Any]]:
    """
    Fetch Sentence nodes that are missing embeddings (or all with text if force_reembed).
    When force_reembed is True, skip is used to paginate (avoid re-fetching the same batch).
    """
    if force_reembed:
        query = """
        MATCH (s:Sentence)
        WHERE s.text IS NOT NULL AND trim(s.text) <> ""
        RETURN s.chunk_id AS chunk_id, s.text AS text
        ORDER BY s.chunk_id
        SKIP $skip
        LIMIT $limit
        """
        return execute_query(query, parameters={"limit": limit, "skip": skip})
    query = """
        MATCH (s:Sentence)
        WHERE s.text IS NOT NULL
          AND trim(s.text) <> ""
          AND (s.embedding IS NULL OR size(s.embedding) = 0)
        RETURN s.chunk_id AS chunk_id, s.text AS text
        LIMIT $limit
        """
    return execute_query(query, parameters={"limit": limit})


def store_embeddings(rows: List[Dict[str, Any]]):
    """
    Store embeddings back into Neo4j.
    """
    query = """
    UNWIND $rows AS row
    MATCH (s:Sentence {chunk_id: row.chunk_id})
    SET s.embedding = row.embedding
    """
    execute_write(query, parameters={"rows": rows})


def embed_batch_with_retry(
    texts: List[str],
    max_retries: int = 5,
) -> List[List[float]]:
    """
    Generate embeddings using Together API with correct endpoint.
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": settings.EMBED_MODEL,
        "input": texts,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                TOGETHER_API_URL,
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            
            # Extract embeddings from response
            # Together API returns: {"data": [{"embedding": [...], "index": 0}, ...]}
            embeddings = [item["embedding"] for item in data["data"]]
            
            return embeddings

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            logger.error(f"Response content: {e.response.text if hasattr(e, 'response') else 'N/A'}")
            
            if attempt == max_retries - 1:
                logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                raise

            base_wait = 5 * (2 ** attempt)
            jitter = random.uniform(0, 2)
            wait_time = min(base_wait + jitter, 60)

            logger.warning(
                f"Embedding error: {e}. "
                f"Retrying in {wait_time:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)

        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Embedding failed after {max_retries} attempts: {e}")
                raise

            base_wait = 5 * (2 ** attempt)
            jitter = random.uniform(0, 2)
            wait_time = min(base_wait + jitter, 60)

            logger.warning(
                f"Embedding error: {e}. "
                f"Retrying in {wait_time:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)

    raise RuntimeError("Embedding failed after all retries")


# ---------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------
def process_embeddings(batch_size: int = None, force_reembed: bool = False):
    """
    Process embeddings for all Sentence nodes missing embeddings.
    If force_reembed is True, re-embed all sentences with text (overwrite existing).
    """
    configured_batch = batch_size or int(BATCH_SIZE)
    effective_batch_size = max(configured_batch, 8)

    total_sentences = count_sentences_to_embed(force_reembed)
    if total_sentences == 0:
        logger.info("No sentences to process.")
        return

    logger.info(
        f"Embedding {total_sentences} sentences "
        f"(batch_size={effective_batch_size}, model={settings.EMBED_MODEL})"
    )

    total_embedded = 0
    pbar = tqdm(total=total_sentences, unit="sent", desc="Embedding")

    while True:
        sentences = fetch_sentences_missing_embeddings(
            limit=effective_batch_size,
            force_reembed=force_reembed,
            skip=total_embedded if force_reembed else 0,
        )

        if not sentences:
            break

        texts = [normalize_text(s["text"]) for s in sentences]
        chunk_ids = [s["chunk_id"] for s in sentences]

        try:
            embeddings = embed_batch_with_retry(texts)

            rows = [
                {"chunk_id": cid, "embedding": emb}
                for cid, emb in zip(chunk_ids, embeddings)
            ]

            store_embeddings(rows)

            total_embedded += len(rows)
            pbar.update(len(rows))

            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Batch failed: {e}")
            time.sleep(5)

    pbar.close()
    logger.info(f"Embedding completed. Total embedded: {total_embedded}")


def main(args):
    try:
        logger.info("=" * 60)
        logger.info("Starting Sentence Embedding Job")
        logger.info(f"API Endpoint   : {TOGETHER_API_URL}")
        logger.info(f"Embedding model: {settings.EMBED_MODEL}")
        logger.info(f"Batch size     : {settings.BATCH_SIZE}")
        logger.info("=" * 60)

        # Quick connectivity test
        logger.info("Testing API connectivity...")
        try:
            test_embeddings = embed_batch_with_retry(["test"])
            logger.info(f"✓ API test successful (embedding dimension: {len(test_embeddings[0])})")
        except Exception as e:
            logger.error(f"✗ API connectivity test failed: {e}")
            raise
        if args.force_reembed:
            logger.info("Re-embedding all sentences (overwriting existing embeddings)")
        process_embeddings(
            batch_size=int(BATCH_SIZE),
            force_reembed=args.force_reembed,
        )

        logger.info("=" * 60)
        logger.info("Embedding job completed successfully")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Embedding job interrupted by user")

    except Exception as e:
        logger.error(f"Embedding job failed: {e}")
        raise

    finally:
        close_driver()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed Sentence nodes using Together AI. By default only processes sentences missing embeddings."
    )
    parser.add_argument(
        "--force",
        "--re-embed",
        dest="force_reembed",
        action="store_true",
        help="Re-embed all sentences with text (overwrite existing embeddings). Use when switching to a different dimension or model (e.g. 1536 -> 768).",
    )
    args = parser.parse_args()
    main(args)