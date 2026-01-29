"""Prompt templates for RAG generation."""
from typing import List, Dict, Any


def build_rag_prompt(query: str, evidence: List[Dict[str, Any]]) -> str:
    """
    Build a strict RAG prompt with citation requirements.
    
    Args:
        query: User query
        evidence: List of retrieved evidence with citation information
        
    Returns:
        Formatted prompt string with strict instructions
    """
    # Format evidence with citations
    evidence_texts = []
    for i, ev in enumerate(evidence, 1):
        doi = ev.get("doi") or ""
        chunk_id = ev.get("chunk_id") or ""
        
        # Build citation: [DOI|chunk_id] if both exist, [chunk_id] if only chunk_id, [unknown] otherwise
        if doi and chunk_id:
            citation = f"[{doi}|{chunk_id}]"
        elif chunk_id:
            citation = f"[{chunk_id}]"
        elif doi:
            citation = f"[{doi}]"
        else:
            citation = "[unknown]"
        
        evidence_texts.append(
            f"[Evidence {i}] {citation}\n{ev.get('sentence', '')}"
        )
    
    evidence_block = "\n\n".join(evidence_texts)
    
    prompt = f"""You are a scientific research assistant. Write a short overall summary that answers the question using ONLY the provided evidence. The evidence will be shown separately below your answer, so do NOT repeat it.

EVIDENCE:
{evidence_block}

INSTRUCTIONS:
1. Write a concise overall summary (2–4 paragraphs) that synthesizes the evidence and directly answers the question.
2. Do NOT list "main findings", bullet points, or quote the evidence verbatim—the evidence block is displayed separately for the user.
3. Use ONLY information from the evidence above. If the evidence is insufficient, state: "Insufficient evidence in corpus."
4. You may use markdown (e.g. **bold**, *italic*, lists, line breaks) for readability.
5. Do not make up information or use knowledge outside the provided evidence.

QUESTION: {query}

ANSWER:"""
    
    return prompt
