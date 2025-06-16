# RAG-Powered Documentation Assistant

## Overview
An intelligent document search and question-answering system using Retrieval-Augmented Generation (RAG) to provide accurate answers from TypeScript documentation using embeddings, similarity search, and LLMs.

## Features
- Semantic document search using vector embeddings
- Cosine similarity-based content retrieval
- Context-aware answer generation with LLMs
- RESTful API with CORS support
- Intelligent document chunking and preprocessing

## Technologies Used
- **Python 3.8+** - Core programming language
- **Flask** - Web framework
- **AIPipe** - LLM and embedding API integration
- **NumPy/Math** - Vector similarity calculations
- **Requests** - HTTP client for API calls

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- AIPipe API token

### Installation
1. Clone the repository:
git clone https://github.com/yourusername/rag-documentation-assistant.git
cd rag-documentation-assistant

text

2. Create virtual environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

text

3. Install dependencies:
pip install -r requirements.txt

text

4. Clone TypeScript documentation:
git clone --depth 1 https://github.com/basarat/typescript-book

text

5. Set environment variable:
export AIPIPE_TOKEN=your_aipipe_token_here

text

### Usage
1. Start the server:
python rag_server.py

text

2. Query the API:
curl "http://localhost:8001/search?q=What%20does%20the%20author%20affectionately%20call%20the%20%3D%3E%20syntax?"

text

## API Endpoints
- **GET /search**: Query documentation with natural language
- Parameter: `q` (query string)
- Returns: JSON with answer, sources, and query

- **GET /health**: Health check endpoint

## Project Structure
rag-documentation-assistant/
├── rag_server.py # Main Flask application
├── requirements.txt # Python dependencies
├── typescript-book/ # TypeScript documentation
├── venv/ # Virtual environment
├── .gitignore # Git ignore rules
└── README.md # This file

text

## Key Achievements
- Implemented advanced document chunking and vector embedding generation
- Developed cosine similarity-based retrieval system
- Integrated multiple AI services for comprehensive document understanding
- Built scalable Flask API with proper error handling
- Optimized keyword matching algorithms for improved accuracy
