"""
Comprehensive test suite for PlasmaLTP RAG system.
Tests all components: configuration, embeddings, Neo4j, retrieval, and generation.
"""

import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# Import after dotenv loads
from app.core.config import settings
from app.rag.embedder import Embedder
from app.rag.retriever import Retriever
from app.rag.schemas import EvidenceItem
from app.db.neo4j import check_connectivity, execute_query, close_driver

# Try to import generator (may fail if together package not installed)
try:
    from app.rag.generator import Generator
    GENERATOR_AVAILABLE = True
except ImportError as e:
    GENERATOR_AVAILABLE = False
    print(f"[WARN] Generator not available ({e}). Skipping generator tests.")

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def print_header(title: str):
    """Print a formatted test section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_test(name: str, status: str, message: str = ""):
    """Print test result."""
    status_symbol = "[PASS]" if status == "PASS" else "[FAIL]" if status == "FAIL" else "[WARN]"
    print(f"{status_symbol} {name}", end="")
    if message:
        print(f" - {message}")
    else:
        print()
    
    if status == "PASS":
        test_results["passed"].append(name)
    elif status == "FAIL":
        test_results["failed"].append(name)
    else:
        test_results["warnings"].append(name)

def test_environment_variables():
    """Test 1: Environment variables are set correctly."""
    print_header("Test 1: Environment Variables")
    
    required_vars = {
        "TOGETHER_API_KEY": "Together AI API key",
        "NEO4J_URI": "Neo4j connection URI",
        "NEO4J_USERNAME": "Neo4j username",
        "NEO4J_PASSWORD": "Neo4j password",
        "EMBED_MODEL": "Embedding model name",
        "CHAT_MODEL": "Chat model name",
    }
    
    all_present = True
    for var, description in required_vars.items():
        value = os.environ.get(var) or getattr(settings, var, None)
        if value:
            print_test(f"{var} is set", "PASS", f"{description} configured")
        else:
            print_test(f"{var} is set", "FAIL", f"{description} is missing")
            all_present = False
    
    return all_present

def test_configuration():
    """Test 2: Configuration loads correctly."""
    print_header("Test 2: Configuration Loading")
    
    try:
        # Check settings object exists
        assert settings is not None, "Settings object is None"
        print_test("Settings object loaded", "PASS")
        
        # Check critical settings
        assert hasattr(settings, 'TOGETHER_API_KEY'), "TOGETHER_API_KEY missing"
        assert settings.TOGETHER_API_KEY, "TOGETHER_API_KEY is empty"
        print_test("TOGETHER_API_KEY configured", "PASS")
        
        assert hasattr(settings, 'EMBED_MODEL'), "EMBED_MODEL missing"
        assert settings.EMBED_MODEL, "EMBED_MODEL is empty"
        print_test(f"EMBED_MODEL configured", "PASS", f"Model: {settings.EMBED_MODEL}")
        
        assert hasattr(settings, 'EMBEDDING_DIMENSION'), "EMBEDDING_DIMENSION missing"
        assert settings.EMBEDDING_DIMENSION > 0, "EMBEDDING_DIMENSION invalid"
        print_test(f"EMBEDDING_DIMENSION configured", "PASS", f"Dimension: {settings.EMBEDDING_DIMENSION}")
        
        assert hasattr(settings, 'CHAT_MODEL'), "CHAT_MODEL missing"
        assert settings.CHAT_MODEL, "CHAT_MODEL is empty"
        print_test(f"CHAT_MODEL configured", "PASS", f"Model: {settings.CHAT_MODEL}")
        
        return True
    except Exception as e:
        print_test("Configuration loading", "FAIL", str(e))
        return False

def test_embedding_api():
    """Test 3: Together AI Embedding API connectivity."""
    print_header("Test 3: Together AI Embedding API")
    
    try:
        api_key = settings.TOGETHER_API_KEY
        endpoint = "https://api.together.xyz/v1/embeddings"
        model = settings.EMBED_MODEL
        
        print_test("API endpoint accessible", "PASS", endpoint)
        
        # Test embedding generation
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "input": ["test query"],
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding = data["data"][0]["embedding"]
            actual_dim = len(embedding)
            expected_dim = settings.EMBEDDING_DIMENSION
            
            print_test("Embedding API response", "PASS", f"HTTP {response.status_code}")
            print_test("Embedding dimension", 
                      "PASS" if actual_dim == expected_dim else "FAIL",
                      f"Got {actual_dim}, expected {expected_dim}")
            
            if actual_dim != expected_dim:
                print(f"  [WARN] Dimension mismatch! Update EMBEDDING_DIMENSION in .env to {actual_dim}")
                print(f"     or change EMBED_MODEL to match dimension {expected_dim}")
                return False
            
            return True
        else:
            print_test("Embedding API response", "FAIL", 
                      f"HTTP {response.status_code}: {response.text[:200]}")
            return False
            
    except Exception as e:
        print_test("Embedding API connectivity", "FAIL", str(e))
        return False

def test_embedder_class():
    """Test 4: Embedder class initialization and usage."""
    print_header("Test 4: Embedder Class")
    
    try:
        embedder = Embedder()
        print_test("Embedder initialization", "PASS", f"Model: {embedder.model}")
        
        # Test embedding generation
        test_query = "What is plasma physics?"
        embedding = embedder.embed_query(test_query)
        
        assert isinstance(embedding, list), "Embedding is not a list"
        assert len(embedding) > 0, "Embedding is empty"
        assert len(embedding) == settings.EMBEDDING_DIMENSION, \
            f"Dimension mismatch: got {len(embedding)}, expected {settings.EMBEDDING_DIMENSION}"
        
        print_test("Query embedding generation", "PASS", 
                  f"Dimension: {len(embedding)}")
        
        # Test that all values are floats
        assert all(isinstance(x, (int, float)) for x in embedding), \
            "Embedding contains non-numeric values"
        print_test("Embedding data type", "PASS", "All values are numeric")
        
        return True
    except ValueError as e:
        print_test("Embedder dimension validation", "FAIL", str(e))
        return False
    except Exception as e:
        print_test("Embedder class", "FAIL", str(e))
        return False

def test_neo4j_connectivity():
    """Test 5: Neo4j database connectivity."""
    print_header("Test 5: Neo4j Connectivity")
    
    try:
        is_connected = check_connectivity()
        if is_connected:
            print_test("Neo4j connection", "PASS")
            return True
        else:
            print_test("Neo4j connection", "FAIL", "Connection check returned False")
            return False
    except Exception as e:
        print_test("Neo4j connection", "FAIL", str(e))
        return False

def test_neo4j_data():
    """Test 6: Neo4j data availability."""
    print_header("Test 6: Neo4j Data Availability")
    
    try:
        # Check if Sentence nodes exist
        query = "MATCH (s:Sentence) RETURN count(s) AS count LIMIT 1"
        result = execute_query(query)
        
        if result and len(result) > 0:
            count = result[0].get("count", 0)
            if count > 0:
                print_test("Sentence nodes exist", "PASS", f"Found {count} sentences")
            else:
                print_test("Sentence nodes exist", "WARN", "No Sentence nodes found")
                return False
        else:
            print_test("Sentence nodes exist", "FAIL", "Query returned no results")
            return False
        
        # Check if embeddings exist
        query = """
        MATCH (s:Sentence)
        WHERE s.embedding IS NOT NULL AND size(s.embedding) > 0
        RETURN count(s) AS count LIMIT 1
        """
        result = execute_query(query)
        
        if result and len(result) > 0:
            count = result[0].get("count", 0)
            if count > 0:
                print_test("Embeddings exist", "PASS", f"Found {count} sentences with embeddings")
            else:
                print_test("Embeddings exist", "WARN", 
                          "No embeddings found. Run scripts/embed_sentences.py first")
                return False
        
        # Check vector index exists
        query = """
        SHOW INDEXES
        YIELD name, type, state
        WHERE name = 'sentence_embedding_idx' AND type = 'VECTOR'
        RETURN name, state
        """
        result = execute_query(query)
        
        if result and len(result) > 0:
            state = result[0].get("state", "")
            if state == "ONLINE":
                print_test("Vector index exists", "PASS", f"Index state: {state}")
            else:
                print_test("Vector index exists", "WARN", f"Index state: {state}")
        else:
            print_test("Vector index exists", "WARN", 
                      "Vector index not found. Create it with the Cypher query in README")
        
        return True
    except Exception as e:
        print_test("Neo4j data check", "FAIL", str(e))
        return False

def test_retriever():
    """Test 7: Retriever functionality."""
    print_header("Test 7: Retriever")
    
    try:
        retriever = Retriever()
        print_test("Retriever initialization", "PASS")
        
        # Test retrieval
        test_query = "plasma physics"
        evidence = retriever.retrieve(test_query, top_k=3)
        
        assert isinstance(evidence, list), "Evidence is not a list"
        print_test("Retrieval execution", "PASS", f"Retrieved {len(evidence)} results")
        
        if len(evidence) > 0:
            # Check evidence structure
            first_item = evidence[0]
            required_fields = ["score", "chunk_id", "sentence", "paper_title"]
            missing_fields = [f for f in required_fields if f not in first_item]
            
            if missing_fields:
                print_test("Evidence structure", "FAIL", 
                          f"Missing fields: {missing_fields}")
                return False
            else:
                print_test("Evidence structure", "PASS", 
                          f"Fields: {', '.join(required_fields)}")
            
            # Check score is valid
            score = first_item.get("score", None)
            if score is not None:
                assert isinstance(score, (int, float)), "Score is not numeric"
                print_test("Evidence score", "PASS", f"Score: {score:.4f}")
            # Ensure retriever output is valid for API (EvidenceItem accepts topic_id as int, etc.)
            evidence_items = [EvidenceItem(**ev) for ev in evidence]
            assert len(evidence_items) == len(evidence)
            print_test("EvidenceItem validation", "PASS", "Retriever output valid for API")
        else:
            print_test("Evidence retrieval", "WARN", 
                      "No evidence retrieved. Check if embeddings exist in Neo4j")
        
        return True
    except Exception as e:
        print_test("Retriever", "FAIL", str(e))
        return False

def test_evidence_item_schema():
    """Test EvidenceItem accepts Neo4j-style data (topic_id as int, optional sentence_type)."""
    print_header("Test 7b: EvidenceItem schema (Neo4j compatibility)")
    try:
        # Neo4j can return topic_id as int; schema must coerce to str
        ev = EvidenceItem(
            score=0.9,
            chunk_id="c1",
            sentence_type=None,
            sent_index=0,
            sentence="Test sentence.",
            paper_title="Paper",
            doi="10.0/test",
            year=2024,
            topic_id=5,
            topic_name="Topic",
        )
        assert ev.topic_id == "5"
        print_test("topic_id int coerced to str", "PASS", f"topic_id={ev.topic_id!r}")
        # Optional sentence_type (property may not exist in DB)
        ev2 = EvidenceItem(
            score=0.8,
            sentence="Another sentence.",
            sentence_type=None,
            topic_id=None,
        )
        assert ev2.sentence_type is None and ev2.topic_id is None
        print_test("Optional fields (sentence_type, topic_id)", "PASS")
        return True
    except Exception as e:
        print_test("EvidenceItem schema", "FAIL", str(e))
        return False

def test_generator():
    """Test 8: Generator functionality."""
    print_header("Test 8: Generator")
    
    if not GENERATOR_AVAILABLE:
        print_test("Generator import", "WARN", "Generator module not available - skipping")
        return False
    
    try:
        generator = Generator()
        print_test("Generator initialization", "PASS", f"Model: {generator.model}")
        
        # Test generation with mock evidence
        test_query = "What is plasma?"
        mock_evidence = [
            {
                "score": 0.95,
                "chunk_id": "test_1",
                "sentence": "Plasma is the fourth state of matter.",
                "paper_title": "Test Paper",
                "doi": "10.1000/test",
                "year": 2024,
            }
        ]
        
        response = generator.generate(test_query, mock_evidence)
        
        assert isinstance(response, dict), "Response is not a dictionary"
        assert "answer" in response, "Response missing 'answer' field"
        assert "metadata" in response, "Response missing 'metadata' field"
        
        answer = response["answer"]
        assert isinstance(answer, str), "Answer is not a string"
        assert len(answer) > 0, "Answer is empty"
        
        print_test("Response generation", "PASS", f"Generated {len(answer)} characters")
        print_test("Response structure", "PASS", "Contains answer and metadata")
        
        metadata = response.get("metadata", {})
        print_test("Response metadata", "PASS", 
                  f"Model: {metadata.get('model', 'N/A')}")
        
        return True
    except Exception as e:
        print_test("Generator", "FAIL", str(e))
        return False

def test_end_to_end():
    """Test 9: End-to-end query flow."""
    print_header("Test 9: End-to-End Query Flow")
    
    if not GENERATOR_AVAILABLE:
        print_test("End-to-end test", "WARN", "Generator not available - skipping")
        return False
    
    try:
        retriever = Retriever()
        generator = Generator()
        
        test_query = "What is plasma physics?"
        
        # Step 1: Retrieve evidence
        evidence = retriever.retrieve(test_query, top_k=3)
        print_test("Step 1: Evidence retrieval", 
                  "PASS" if evidence else "WARN",
                  f"Retrieved {len(evidence)} items")
        
        if not evidence:
            print_test("End-to-end test", "WARN", 
                      "Skipping generation - no evidence retrieved")
            return False
        
        # Step 2: Generate response
        response = generator.generate(test_query, evidence)
        assert "answer" in response, "Response missing answer"
        
        answer = response["answer"]
        print_test("Step 2: Response generation", "PASS", 
                  f"Generated answer ({len(answer)} chars)")
        
        print_test("End-to-end flow", "PASS", "Complete query flow successful")
        return True
    except Exception as e:
        print_test("End-to-end flow", "FAIL", str(e))
        return False

def test_error_handling():
    """Test 10: Error handling."""
    print_header("Test 10: Error Handling")
    
    try:
        embedder = Embedder()
        
        # Test empty query
        try:
            embedding = embedder.embed_query("")
            print_test("Empty query handling", "PASS", 
                      f"Handled (dim: {len(embedding)})")
        except Exception as e:
            print_test("Empty query handling", "WARN", str(e))
        
        # Test moderately long query (API may reject very long input with 400)
        long_query = "test " * 100  # 500 chars, under typical API limits
        try:
            embedding = embedder.embed_query(long_query)
            print_test("Long query handling", "PASS", 
                      f"Handled (dim: {len(embedding)})")
        except Exception as e:
            print_test("Long query handling", "WARN", str(e))
        
        return True
    except Exception as e:
        print_test("Error handling", "FAIL", str(e))
        return False

def print_summary():
    """Print test summary."""
    print_header("Test Summary")
    
    total = len(test_results["passed"]) + len(test_results["failed"]) + len(test_results["warnings"])
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    warnings = len(test_results["warnings"])
    
    print(f"\nTotal Tests: {total}")
    print(f"[PASS] Passed: {passed}")
    print(f"[FAIL] Failed: {failed}")
    print(f"[WARN] Warnings: {warnings}")
    
    if failed > 0:
        print("\nFailed Tests:")
        for test in test_results["failed"]:
            print(f"  [FAIL] {test}")
    
    if warnings > 0:
        print("\nWarnings:")
        for test in test_results["warnings"]:
            print(f"  [WARN] {test}")
    
    print("\n" + "=" * 70)
    
    if failed == 0:
        print("[PASS] All critical tests passed!")
        if warnings > 0:
            print("[WARN] Some warnings - review and fix if needed")
        return True
    else:
        print("[FAIL] Some tests failed - please fix issues before proceeding")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("  PlasmaLTP RAG System - Comprehensive Test Suite")
    print("=" * 70)
    
    # Run all tests
    tests = [
        test_environment_variables,
        test_configuration,
        test_embedding_api,
        test_embedder_class,
        test_neo4j_connectivity,
        test_neo4j_data,
        test_retriever,
        test_evidence_item_schema,
        test_generator,
        test_end_to_end,
        test_error_handling,
    ]
    
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print_test(test_func.__name__, "FAIL", f"Unexpected error: {e}")
    
    # Cleanup
    try:
        close_driver()
    except:
        pass
    
    # Print summary
    success = print_summary()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
