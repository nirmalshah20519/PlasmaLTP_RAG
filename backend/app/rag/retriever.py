"""Document retrieval functionality."""
from typing import List, Dict, Any, Optional
from app.db.neo4j import execute_query
from app.rag.embedder import Embedder
from app.core.config import settings
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def _query_vector_for_neo4j(embedding: List[float], expected_dim: int):
    """
    Return the query vector in the format Neo4j expects for db.index.vector.queryNodes.

    Exact error: "Index query vector has a dimensionality of 768, but provided vector has 1536."
    Root cause: Neo4j Aura (and some 5.x) treats the query vector as float32. When we send
    768 Python floats they are serialized as 64-bit (6144 bytes); the procedure then
    interprets 6144/4 = 1536 dimensions. Sending neo4j.vector.Vector(f32) still yields 1536
    on current Aura (server may convert Vector to 64-bit list internally).

    Workaround: use a 1536-dimensional index and embedding model. Then we send a list of
    1536 floats; the procedure uses list length as dimension, so 1536 matches.
    """
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Query embedding dimension {len(embedding)} does not match "
            f"EMBEDDING_DIMENSION {expected_dim}."
        )
    # 1536 workaround: send plain list so procedure sees 1536 elements (list length = dimension)
    if expected_dim == 1536:
        return [float(x) for x in embedding[:expected_dim]]
    # For 768: try Vector(f32) in case the server supports Bolt 6 and accepts it correctly
    try:
        from neo4j.vector import Vector
        import numpy as np
        arr = np.array(embedding[:expected_dim], dtype=np.float32)
        return Vector(arr)
    except Exception as e:
        from neo4j.exceptions import ConfigurationError
        if isinstance(e, ConfigurationError) and "Bolt" in str(e):
            raise RuntimeError(
                "Neo4j vector search with 768 dimensions requires Bolt 6 (Neo4j 2025.10+). "
                "Your server uses Bolt 5. Use the 1536 workaround: set EMBEDDING_DIMENSION=1536, "
                "use a 1536-dim embedding model (e.g. text-embedding-3-small), recreate the vector "
                "index with 1536 dimensions, re-run embed_sentences.py, then run again."
            ) from e
        try:
            from neo4j.vector import Vector
            return Vector(embedding[:expected_dim], "f32")
        except Exception:
            pass
        return [float(x) for x in embedding[:expected_dim]]


class Retriever:
    """Handles retrieval of relevant documents from Neo4j using vector search."""
    
    def __init__(self):
        """Initialize the retriever with embedder."""
        self.embedder = Embedder()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant evidence for a query using Neo4j vector search.
        
        Args:
            query: Query text
            top_k: Number of results to return (defaults to config value)
            
        Returns:
            List of evidence dictionaries with all required fields
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        expected_dim = settings.EMBEDDING_DIMENSION
        
        qvec = _query_vector_for_neo4j(query_embedding, expected_dim)
        
        # Execute vector search with Paper and Topic expansion
        cypher_query = """
        CALL db.index.vector.queryNodes('sentence_embedding_idx', $k, $qvec)
        YIELD node AS s, score
        MATCH (p:Paper)-[:HAS_SENTENCE]->(s)
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
        RETURN 
            score,
            s.chunk_id AS chunk_id,
            s.sentence_type AS sentence_type,
            s.sent_index AS sent_index,
            s.text AS sentence,
            p.title AS paper_title,
            p.doi AS doi,
            p.year AS year,
            t.topic_id AS topic_id,
            t.topic_name AS topic_name
        ORDER BY score DESC
        """
        try:
            results = execute_query(
                cypher_query,
                parameters={
                    "k": top_k,
                    "qvec": qvec
                }
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "bolt" in err_msg and "6" in err_msg and "vector" in err_msg:
                raise RuntimeError(
                    "Neo4j vector search requires Bolt 6 (Neo4j 2025.10+). "
                    "Your server uses Bolt 5. Options: (1) Upgrade Neo4j/Aura to support Bolt 6, or "
                    "(2) Recreate the vector index with 1536 dimensions, set EMBEDDING_DIMENSION=1536, "
                    "use a 1536-dim embedding model, re-embed data, then run again."
                ) from e
            if "1536" in err_msg and "768" in err_msg:
                raise RuntimeError(
                    "Neo4j reports 'provided vector has 1536' (index expects 768). "
                    "Use the 1536 workaround: set EMBEDDING_DIMENSION=1536, use a 1536-dim embedding "
                    "model, recreate the vector index with 1536 dimensions, re-run embed_sentences.py. "
                    "See README section 'Error: Index query vector has a dimensionality of 768, but provided vector has 1536'."
                ) from e
            raise
        
        # Format results as evidence
        evidence = []
        for result in results:
            evidence.append({
                "score": result.get("score", 0.0),
                "chunk_id": result.get("chunk_id"),
                "sentence_type": result.get("sentence_type"),
                "sent_index": result.get("sent_index"),
                "sentence": result.get("sentence", ""),
                "paper_title": result.get("paper_title"),
                "doi": result.get("doi"),
                "year": result.get("year"),
                "topic_id": result.get("topic_id"),
                "topic_name": result.get("topic_name")
            })
        
        return evidence
    
    def merge_evidence_by_doi(
        self,
        evidence: List[Dict[str, Any]],
        max_items: int = 3,
        sentence_sep: str = " ",
    ) -> List[Dict[str, Any]]:
        """
        Group evidence by DOI, concatenate sentences within each group, then return
        the top max_items by best score so the response has up to max_items distinct papers.
        When multiple results share the same DOI, their sentences are merged into one.
        """
        if not evidence:
            return []
        # Group by DOI (treat None/empty as same key for "no DOI")
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for item in evidence:
            doi = (item.get("doi") or "").strip() or "_no_doi"
            groups.setdefault(doi, []).append(item)
        # Build one merged item per DOI: concatenate sentences, keep best score and metadata
        merged_list: List[Dict[str, Any]] = []
        for doi_key, items in groups.items():
            # Sort by score descending so best is first
            sorted_items = sorted(items, key=lambda x: x.get("score", 0.0), reverse=True)
            best = sorted_items[0]
            sentences = [it.get("sentence", "").strip() for it in sorted_items if it.get("sentence")]
            combined_sentence = sentence_sep.join(sentences) if sentences else best.get("sentence", "")
            merged_list.append({
                "score": best.get("score", 0.0),
                "chunk_id": best.get("chunk_id"),
                "sentence_type": best.get("sentence_type"),
                "sent_index": best.get("sent_index"),
                "sentence": combined_sentence,
                "paper_title": best.get("paper_title"),
                "doi": best.get("doi") if doi_key != "_no_doi" else None,
                "year": best.get("year"),
                "topic_id": best.get("topic_id"),
                "topic_name": best.get("topic_name"),
            })
        # Sort merged by score descending and return top max_items
        merged_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return merged_list[:max_items]
    
    def deduplicate_and_filter_evidence(
        self, 
        evidence: List[Dict[str, Any]], 
        max_per_doi: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate evidence by (doi, chunk_id) and limit max sentences per DOI.
        Prefer sentence_type == "claim" if present (rerank: claim first).
        
        Args:
            evidence: List of evidence dictionaries
            max_per_doi: Maximum number of sentences per DOI
            
        Returns:
            Filtered and deduplicated evidence list
        """
        if not evidence:
            return []
        
        # Sort by score descending, then prioritize "claim" sentence_type
        def sort_key(item: Dict[str, Any]) -> tuple:
            score = item.get("score", 0.0)
            sentence_type = item.get("sentence_type", "")
            # Claims get priority boost (multiply score by 1.1 if claim)
            priority_boost = 1.1 if sentence_type == "claim" else 1.0
            return (-score * priority_boost, sentence_type != "claim")
        
        sorted_evidence = sorted(evidence, key=sort_key)
        
        # Deduplicate by (doi, chunk_id) and limit per DOI
        seen = set()
        doi_counts: Dict[str, int] = {}
        filtered_evidence = []
        
        for item in sorted_evidence:
            doi = item.get("doi") or ""
            chunk_id = item.get("chunk_id") or ""
            key = (doi, chunk_id)
            
            # Skip if already seen
            if key in seen:
                continue
            
            # Check DOI limit
            if doi:
                count = doi_counts.get(doi, 0)
                if count >= max_per_doi:
                    continue
                doi_counts[doi] = count + 1
            
            seen.add(key)
            filtered_evidence.append(item)
        
        return filtered_evidence
