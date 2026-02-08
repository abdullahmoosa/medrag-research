#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Backend API Server for Medical RAG Chat Application
This server provides REST API endpoints for medical question answering using RAG.
Frontend clients can connect from other machines to use this service.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.rag_service import RAGService
from src.app.config import RAGConfig
from src.app.models import ChatRequest, ChatResponse, ConfigRequest
from src.app.conversation_manager import conversation_manager

app = FastAPI(
    title="Medical RAG Backend API",
    description="Backend API for medical question answering using RAG - runs on GPU server",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc
)

# Enable CORS for frontend connections from other machines
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify specific frontend domains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service: Optional[RAGService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG service on startup"""
    global rag_service
    print("🚀 Initializing Medical RAG Backend Service...")
    config = RAGConfig()
    rag_service = RAGService(config)
    await rag_service.initialize()
    print("✅ Medical RAG Backend Service ready!")
    print(f"   - Index: {config.index_name}")
    print(f"   - Embedding Model: {config.embed_model}")
    print(f"   - LLM Model: {config.llm_model}")
    print(f"   - Device: {config.device}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, client_request: Request):
    """
    Handle chat requests with RAG and conversation history
    
    This endpoint processes user questions and returns evidence-based answers
    with source citations from medical literature. Maintains conversation history.
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        # Get client IP for session tracking
        client_ip = client_request.client.host
        
        # Add user message to conversation history
        conversation_id = conversation_manager.add_message(
            conversation_id=request.conversation_id,
            role="user",
            content=request.message,
            user_ip=client_ip
        )
        
        # Get conversation context for LLM
        context = conversation_manager.get_context_for_llm(conversation_id, max_context=6)
        
        # Generate response with context
        response = await rag_service.generate_response(
            query=request.message,
            conversation_id=conversation_id,
            conversation_context=context
        )
        
        # Add assistant response to conversation history
        conversation_manager.add_message(
            conversation_id=conversation_id,
            role="assistant", 
            content=response.response,
            sources=response.sources,
            user_ip=client_ip
        )
        
        # Update response with actual conversation ID
        response.conversation_id = conversation_id
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/api/config")
async def update_config(request: ConfigRequest):
    """
    Update RAG configuration
    
    Allows dynamic reconfiguration of retrieval parameters, models, etc.
    """
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        await rag_service.update_config(request.dict(exclude_unset=True))
        return {"status": "success", "message": "Configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@app.get("/api/config")
async def get_config():
    """Get current RAG configuration"""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    return {
        "status": "success",
        "config": rag_service.config.to_dict()
    }

@app.get("/api/conversations")
async def list_conversations(limit: int = 20):
    """List recent conversations"""
    try:
        sessions = conversation_manager.list_sessions(limit)
        return {
            "status": "success",
            "conversations": sessions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing conversations: {str(e)}")

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, max_messages: int = 50):
    """Get conversation history"""
    try:
        history = conversation_manager.get_conversation_history(conversation_id, max_messages)
        messages = []
        for msg in history:
            messages.append({
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "sources": msg.sources
            })
        
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "messages": messages
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        success = conversation_manager.delete_session(conversation_id)
        if success:
            return {"status": "success", "message": "Conversation deleted"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")

@app.post("/api/conversations/new")
async def create_conversation(client_request: Request):
    """Create a new conversation session"""
    try:
        client_ip = client_request.client.host
        conversation_id = conversation_manager.create_session(client_ip)
        return {
            "status": "success",
            "conversation_id": conversation_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "service": "medical-rag-backend",
        "initialized": rag_service is not None
    }
    
    if rag_service:
        status.update({
            "index": rag_service.config.index_name,
            "embed_model": rag_service.config.embed_model,
            "llm_model": rag_service.config.llm_model,
            "device": rag_service.config.device
        })
    
    return status

@app.get("/api/available-indexes")
async def get_available_indexes():
    """Get list of available indexes"""
    indexes_dir = os.path.join(PROJECT_ROOT, "indexes")
    available_indexes = []
    
    if os.path.exists(indexes_dir):
        for item in os.listdir(indexes_dir):
            index_path = os.path.join(indexes_dir, item)
            if os.path.isdir(index_path):
                # Check if index has required files
                required_files = ["faiss.index", "meta.json"]
                has_required = all(
                    os.path.exists(os.path.join(index_path, f)) 
                    for f in required_files
                )
                
                available_indexes.append({
                    "name": item,
                    "path": index_path,
                    "valid": has_required
                })
    
    return {"indexes": available_indexes}

@app.get("/api/stats")
async def get_service_stats():
    """Get service statistics and performance metrics"""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    # Get index statistics
    index_stats = {}
    if rag_service.index:
        index_stats = {
            "total_documents": len(rag_service.index.df) if hasattr(rag_service.index, 'df') else 0,
            "index_type": "hybrid" if hasattr(rag_service.index, 'faiss_index') and hasattr(rag_service.index, 'bm25_index') else "unknown"
        }
    
    return {
        "service_status": "running",
        "configuration": rag_service.config.to_dict(),
        "index_stats": index_stats,
        "supported_features": [
            "hybrid_retrieval",
            "dense_retrieval", 
            "bm25_retrieval",
            "medembed_embeddings",
            "evidence_packing",
            "source_citations"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Medical RAG Backend API",
        "version": "1.0.0",
        "description": "Backend API for medical question answering using RAG",
        "endpoints": {
            "chat": "/api/chat",
            "config": "/api/config", 
            "health": "/api/health",
            "indexes": "/api/available-indexes",
            "stats": "/api/stats",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "frontend": "Connect from separate frontend applications",
        "cors": "Enabled for cross-origin requests"
    }

# Development/testing endpoints
@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify backend connectivity"""
    return {
        "message": "Backend API is working!",
        "timestamp": "2025-08-17",
        "service": "medical-rag-backend"
    }

if __name__ == "__main__":
    import uvicorn
    print("🏥 Starting Medical RAG Backend API Server")
    print("=" * 50)
    print("This server provides RAG capabilities for frontend clients.")
    print("Frontends can connect from other machines via HTTP API.")
    print("=" * 50)
    
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,
        log_level="info"
    )
