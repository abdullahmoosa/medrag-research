## Project Description

**MedRAG Research** is a comprehensive research framework for medical question-answering systems powered by Retrieval-Augmented Generation (RAG). It combines advanced retrieval techniques with medical-domain-specific language models to deliver evidence-based answers grounded in medical literature.

### Core Components

The project consists of three integrated modules:

1. **Medical RAG Chat Application** - A production-ready FastAPI web application featuring a real-time chat interface with medical question answering, customizable RAG pipelines, and configurable retrieval modes (hybrid, dense, BM25).

2. **RAG Evaluation System** - A modular evaluation framework supporting dual evaluation modes (No-RAG baseline and full RAG), multi-stage retrieval, query variants (reformulation, HyDE, option-aware queries), and comprehensive metrics calculation for scientific experimentation.

3. **Medical Text Rechunking Pipeline** - A configurable data processing pipeline that converts raw medical textbooks into semantically coherent chunks with rich medical metadata (content type, medical systems, keywords), supporting both BM25 sparse and FAISS dense indexing.

### Key Technologies

- **Languages**: Python, JavaScript/HTML (for web UI)
- **Frameworks**: FastAPI, Sentence Transformers, FAISS
- **Integration**: Ollama LLM server, medical-specific embeddings (MedEmbed)
- **Medical Models**: LLaMA3-Med42-8B, DeepSeek-R1, medical-optimized embeddings

### Research Focus

The framework enables systematic research into:
- Multi-stage retrieval pipelines (coarse and fine-grained)
- Query reformulation and expansion strategies
- Hybrid retrieval fusion techniques
- LLM-based answer generation with source attribution
- Evaluation benchmarking against medical question datasets

### Architecture Highlights

- **Modular Design**: Independent, testable components that can be used separately or in combination
- **Type-Safe**: Strong typing throughout with Pydantic models and dataclasses
- **Configuration-Driven**: All experiments controlled via configuration without code changes
- **Research-Ready**: Built for serious experimentation with support for ablation studies and comparative analysis
- **Evidence-Based**: All responses include source citations and relevance scores from medical literature

This is a research-grade project suitable for academic investigations into medical AI, information retrieval, and knowledge-grounded question-answering systems.

## Citation

If you use MedRAG Research in your work, please cite our paper:

```bibtex
@article{sultana2026medrag,
  title={A Systematic Study of Retrieval Pipeline Design for Retrieval-Augmented Medical Question Answering},
  author={Sultana, Nusrat and Moosa, Abdullah Muhammad and Rahman, Kazi Afzalur and Banik, Sajal Chandra},
  journal={arXiv preprint arXiv:2604.07274},
  year={2026}
}
