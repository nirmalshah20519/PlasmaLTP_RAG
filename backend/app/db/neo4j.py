"""Neo4j database connection and utilities."""
from neo4j import GraphDatabase, Driver
from typing import Optional, List, Dict, Any
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Singleton driver instance
_driver: Optional[Driver] = None


def get_driver() -> Driver:
    """
    Get or create the Neo4j driver instance (singleton pattern).
    
    Returns:
        Neo4j driver instance
    """
    global _driver
    
    if _driver is None:
        try:
            logger.info("Initializing Neo4j driver connection")
            _driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            # Verify connectivity
            _driver.verify_connectivity()
            logger.info("Neo4j driver connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise ConnectionError(f"Failed to connect to Neo4j: {str(e)}")
    
    return _driver


def close_driver():
    """Safely close the Neo4j driver connection."""
    global _driver
    
    if _driver is not None:
        try:
            logger.info("Closing Neo4j driver connection")
            _driver.close()
            _driver = None
            logger.info("Neo4j driver closed successfully")
        except Exception as e:
            logger.error(f"Error closing Neo4j driver: {str(e)}")


def execute_query(query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query and return results.
    
    Args:
        query: Cypher query string
        parameters: Optional query parameters
        
    Returns:
        List of result records as dictionaries
    """
    if parameters is None:
        parameters = {}
    
    driver = get_driver()
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        result = session.run(query, parameters)
        return [record.data() for record in result]


def execute_write(query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Execute a write transaction.
    
    Args:
        query: Cypher query string
        parameters: Optional query parameters
        
    Returns:
        List of result records as dictionaries
    """
    if parameters is None:
        parameters = {}
    
    driver = get_driver()
    with driver.session(database=settings.NEO4J_DATABASE) as session:
        def work(tx):
            result = tx.run(query, parameters)
            return list(result)
        records = session.execute_write(work)
        return [record.data() for record in records]


def check_connectivity() -> bool:
    """
    Check Neo4j connectivity by running a simple query.
    
    Returns:
        True if connected, False otherwise
    """
    try:
        driver = get_driver()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            return record is not None and record["test"] == 1
    except Exception as e:
        logger.error(f"Neo4j connectivity check failed: {str(e)}")
        return False

