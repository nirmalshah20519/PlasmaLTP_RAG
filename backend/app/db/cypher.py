"""Cypher query utilities for Neo4j."""
from typing import List, Dict, Any, Optional


def create_sentence_node_query(text: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate Cypher query to create a sentence node with embedding.
    
    Args:
        text: Sentence text
        embedding: Embedding vector
        metadata: Optional metadata dictionary
        
    Returns:
        Cypher query string
    """
    metadata_str = ""
    if metadata:
        metadata_props = ", ".join([f"{k}: ${k}" for k in metadata.keys()])
        metadata_str = f", {metadata_props}"
    
    query = f"""
    CREATE (s:Sentence {{
        text: $text,
        embedding: $embedding{metadata_str}
    }})
    RETURN s
    """
    return query


def similarity_search_query(query_embedding: List[float], top_k: int = 5) -> str:
    """
    Generate Cypher query for vector similarity search.
    
    Note: This function is deprecated. Use the query in retriever.py instead.
    
    Args:
        query_embedding: Query embedding vector
        top_k: Number of results to return
        
    Returns:
        Cypher query string
    """
    query = f"""
    CALL db.index.vector.queryNodes('sentence_embedding_idx', $top_k, $query_embedding)
    YIELD node, score
    RETURN node.text AS text, node.embedding AS embedding, score
    ORDER BY score DESC
    LIMIT $top_k
    """
    return query


def create_index_query() -> str:
    """
    Generate Cypher query to create vector index for embeddings.
    
    Returns:
        Cypher query string
    """
    query = """
    CREATE VECTOR INDEX sentence_embedding_idx IF NOT EXISTS
    FOR (s:Sentence)
    ON s.embedding
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 768,
            `vector.similarity_function`: 'cosine'
        }
    }
    """
    return query


def get_all_sentences_query(limit: Optional[int] = None) -> str:
    """
    Generate Cypher query to retrieve all sentences.
    
    Args:
        limit: Optional limit on number of results
        
    Returns:
        Cypher query string
    """
    limit_clause = f"LIMIT {limit}" if limit else ""
    query = f"""
    MATCH (s:Sentence)
    RETURN s.text AS text, s.embedding AS embedding
    {limit_clause}
    """
    return query

