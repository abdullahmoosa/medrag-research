"""
Medical RAG Chat Application

A FastAPI-based chat application for medical question answering using 
Retrieval-Augmented Generation (RAG) with evidence-based responses.
"""

__version__ = "1.0.0"
__author__ = "MedRAG Research Team"

from .config import RAGConfig
from .models import ChatRequest, ChatResponse, Source
from .rag_service import RAGService

__all__ = [
    "RAGConfig",
    "ChatRequest", 
    "ChatResponse",
    "Source",
    "RAGService"
]
