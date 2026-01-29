# PlasmaLTP RAG

A production-ready Retrieval-Augmented Generation (RAG) system for scientific literature, built with FastAPI and Neo4j.

## Features

- 🔍 **Vector Search**: Semantic search over scientific papers using Neo4j vector indexes
- 🤖 **LLM Integration**: Together AI for embeddings and response generation
- 📊 **Evidence Merging**: Automatically merges multiple results from the same paper
- 🎨 **Modern UI**: Clean, responsive frontend with markdown support
- ⚡ **Fast & Scalable**: Efficient batch processing and resumable embedding jobs

## Prerequisites

- Python 3.8+
- Neo4j Aura instance (or Neo4j 5.13+)
- Together AI API key ([Get one here](https://api.together.xyz/settings/api-keys))

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/nirmalshah20519/PlasmaLTP_RAG.git
cd PlasmaLTP_RAG
```

### 2. Set up Python environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password-here
NEO4J_DATABASE=neo4j

# Together AI Configuration
TOGETHER_API_KEY=your-together-api-key-here
EMBED_MODEL=BAAI/bge-base-en-v1.5
CHAT_MODEL=CHAT_MODEL=mistralai/Mistral-7B-Instruct-v0.3
EMBEDDING_DIMENSION=768

# Application Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False
LOG_LEVEL=INFO

# RAG Configuration
TOP_K_RESULTS=5
EMBEDDING_DIMENSION=768
BATCH_SIZE=128
```

## Setup

### 1. Create Neo4j Vector Index

Open your Neo4j Browser and run:

```cypher
CREATE VECTOR INDEX sentence_embedding_idx IF NOT EXISTS
FOR (s:Sentence) ON s.embedding
OPTIONS {
    indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }
}
```

**Note**: Set `vector.dimensions` to match your `EMBEDDING_DIMENSION` in `.env`.

### 2. Generate Embeddings

Embed all sentences in your Neo4j database:

```bash
python scripts/embed_sentences.py
```

To re-embed all sentences (e.g., after changing the model):

```bash
python scripts/embed_sentences.py --force
```

## Usage

### Start the Backend Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### Open the Frontend

Simply open `frontend/index.html` in your web browser. The frontend connects to the backend at `http://localhost:8000`.

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /query` - Main RAG query endpoint

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings about plasma physics?"}'
```

## Project Structure

```
PlasmaLTP_RAG/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── core/                # Configuration
│   │   ├── db/                  # Neo4j utilities
│   │   └── rag/                 # RAG components
│   ├── scripts/
│   │   └── embed_sentences.py   # Embedding script
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
└── README.md
```

## Configuration

Key environment variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBED_MODEL` | Embedding model name | `BAAI/bge-base-en-v1.5` |
| `EMBEDDING_DIMENSION` | Vector dimension | `768` |
| `CHAT_MODEL` | LLM model for generation | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` |
| `TOP_K_RESULTS` | Number of evidence items | `5` |
| `BATCH_SIZE` | Embedding batch size | `128` |
