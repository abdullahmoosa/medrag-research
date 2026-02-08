# Medical RAG Chat Application 🏥

A FastAPI-based chat application that uses Retrieval-Augmented Generation (RAG) to answer medical questions with evidence-based responses.

## Features

- **Interactive Web Interface**: Clean, modern chat UI with real-time responses
- **Evidence-Based Answers**: All responses include source citations from medical literature
- **Customizable RAG Pipeline**: 
  - Multiple retrieval modes (Hybrid, Dense, BM25)
  - Configurable parameters (top-k, token limits, etc.)
  - Multiple embedding models support
- **Medical Domain Optimization**: Uses MedEmbed and medical-specific LLMs
- **Real-time Configuration**: Update RAG settings without restarting the service
- **Source Transparency**: Shows retrieved passages with relevance scores

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web UI        │───▶│   FastAPI        │───▶│   RAG Service   │
│   (HTML/JS)     │    │   Backend        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────┐              │
                       │   Ollama LLM    │◀─────────────┤
                       │   Server        │              │
                       └─────────────────┘              │
                                                         │
                       ┌─────────────────┐              │
                       │   MedEmbed      │◀─────────────┤
                       │   Embeddings    │              │
                       └─────────────────┘              │
                                                         │
                       ┌─────────────────┐              │
                       │   Hybrid Index  │◀─────────────┘
                       │   (FAISS+BM25)  │
                       └─────────────────┘
```

## Prerequisites

1. **Python 3.8+** with required packages
2. **Ollama** server running locally (default: `localhost:11434`)
3. **Pre-built indexes** in the `indexes/` directory
4. **LLM models** downloaded in Ollama

### Required Ollama Models

```bash
# Download medical LLM models
ollama pull thewindmom/llama3-med42-8b
ollama pull deepseek-ai/deepseek-r1:8b

# Optional: embedding models (if not using MedEmbed)
ollama pull oscardp96/medcpt-query:latest
```

## Installation

1. **Install dependencies:**
   ```bash
   cd src/app
   pip install -r requirements.txt
   ```

2. **Verify indexes exist:**
   ```bash
   # Check that you have indexes built
   ls ../../indexes/
   # Should show directories like: medcorp_medembed, medembed, etc.
   ```

3. **Test Ollama connection:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

## Usage

### 1. Start the Web Application

```bash
# From src/app directory
python run.py

# Or with custom options
python run.py --host 0.0.0.0 --port 8080 --reload
```

Then open your browser to: http://localhost:8000

### 2. CLI Testing (Optional)

Test the RAG service directly from command line:

```bash
# Basic test
python test_cli.py

# With custom parameters
python test_cli.py --index medcorp_medembed --mode hybrid --k 15
```

### 3. API Endpoints

The service exposes several REST endpoints:

- `GET /` - Web interface
- `POST /chat` - Chat with RAG
- `POST /config` - Update configuration
- `GET /health` - Health check
- `GET /available-indexes` - List available indexes

#### Example API Usage:

```bash
# Send a chat message
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of diabetes?", "conversation_id": "test"}'

# Update configuration
curl -X POST "http://localhost:8000/config" \
  -H "Content-Type: application/json" \
  -d '{"mode": "dense", "k": 20, "llm_model": "deepseek-ai/deepseek-r1:8b"}'
```

## Configuration

### Default Settings

- **Index**: `medcorp_medembed`
- **Retrieval Mode**: `hybrid` (FAISS + BM25)
- **Top-K**: 12 passages
- **Embedding Model**: `abhinand/MedEmbed-large-v0.1`
- **LLM Model**: `thewindmom/llama3-med42-8b`
- **Max Evidence Tokens**: 1200

### Customizable Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `index_name` | Which index to use | `medcorp_medembed`, `medembed`, etc. |
| `mode` | Retrieval strategy | `hybrid`, `dense`, `bm25` |
| `k` | Number of passages to retrieve | 1-50 |
| `dense_k` | FAISS results for hybrid mode | 10-200 |
| `bm25_k` | BM25 results for hybrid mode | 50-1000 |
| `max_evidence_tokens` | Token budget for evidence | 200-3000 |
| `llm_model` | Ollama model name | Any compatible model |
| `temperature` | Generation randomness | 0.0-2.0 |

### Environment Variables

```bash
# Optional: Override Ollama URL
export OLLAMA_BASE_URL="http://localhost:11434"

# Optional: CUDA device selection
export CUDA_VISIBLE_DEVICES="0"
```

## Available Indexes

The application supports multiple pre-built indexes:

- **`medcorp_medembed`**: MedCorp corpus with MedEmbed embeddings
- **`medembed`**: General medical literature with MedEmbed
- **`textbooks_medembed_flat`**: Medical textbooks with MedEmbed

## Web Interface Features

### Chat Interface
- Real-time messaging with medical assistant
- Source citations for all answers
- Conversation history
- Responsive design

### Configuration Panel
- Live parameter adjustment
- Index switching
- Model selection
- Performance monitoring

### Source Display
- Relevance scores
- Document titles and sources
- Text snippets
- URL links (when available)

## Performance

Typical response times:
- **Retrieval**: 0.1-0.5 seconds
- **Generation**: 1-5 seconds (depending on model and length)
- **Total**: 1-6 seconds per query

Memory usage:
- **Base**: ~2GB (indexes + embeddings)
- **Peak**: ~4-8GB (during generation)

## Troubleshooting

### Common Issues

1. **"RAG service not initialized"**
   - Check that index directory exists
   - Verify Ollama is running
   - Check model availability

2. **Slow responses**
   - Use GPU acceleration: `--device cuda`
   - Reduce `k` parameter
   - Use smaller LLM models

3. **Empty responses**
   - Check Ollama model compatibility
   - Verify index has data
   - Try different retrieval mode

4. **CUDA out of memory**
   - Use `--device cpu`
   - Reduce batch sizes
   - Use smaller models

### Logging

Enable debug logging:
```bash
python run.py --log-level debug
```

Check Ollama logs:
```bash
ollama logs
```

## Development

### Project Structure

```
src/app/
├── main.py              # FastAPI application
├── rag_service.py       # Core RAG logic
├── config.py           # Configuration management
├── models.py           # Pydantic models
├── run.py              # Startup script
├── test_cli.py         # CLI testing tool
├── requirements.txt    # Dependencies
└── README.md          # This file
```

### Adding New Features

1. **New embedding models**: Update `rag_service.py` initialization
2. **Custom retrievers**: Extend `HybridIndex` class
3. **New LLM backends**: Add providers to `_generate_response()`
4. **UI improvements**: Modify HTML template in `main.py`

### Testing

```bash
# Test configuration validation
python -c "from src.app.config import RAGConfig; RAGConfig().validate()"

# Test service initialization  
python test_cli.py --index medcorp_medembed

# Test API endpoints
curl http://localhost:8000/health
```

## License

This project follows the same license as the parent MedRAG project.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

For more information, see the main project documentation or open an issue on GitHub.
