#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pydantic models for API requests and responses
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User's message/question")
    conversation_id: str = Field(default="default", description="Conversation identifier")
    
class Source(BaseModel):
    """Source document model"""
    doc_id: str = Field(..., description="Document ID")
    title: str = Field(default="", description="Document title")
    text: str = Field(..., description="Document text content")
    source: str = Field(default="", description="Source name")
    score: float = Field(default=0.0, description="Relevance score")
    url: str = Field(default="", description="Source URL if available")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="Generated response")
    sources: List[Source] = Field(default_factory=list, description="Retrieved sources")
    conversation_id: str = Field(..., description="Conversation identifier")
    num_sources: int = Field(default=0, description="Number of sources retrieved")
    retrieval_time: float = Field(default=0.0, description="Time spent on retrieval (seconds)")
    generation_time: float = Field(default=0.0, description="Time spent on generation (seconds)")

class ConfigRequest(BaseModel):
    """Request model for configuration updates"""
    index_name: Optional[str] = Field(None, description="Name of the index to use")
    mode: Optional[str] = Field(None, description="Retrieval mode (hybrid, dense, bm25)")
    k: Optional[int] = Field(None, description="Number of top results to retrieve")
    dense_k: Optional[int] = Field(None, description="Number of dense results for hybrid mode")
    bm25_k: Optional[int] = Field(None, description="Number of BM25 results for hybrid mode")
    max_evidence_tokens: Optional[int] = Field(None, description="Maximum tokens for evidence")
    llm_model: Optional[str] = Field(None, description="LLM model to use")
    temperature: Optional[float] = Field(None, description="Temperature for generation")
    
class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    
class IndexListResponse(BaseModel):
    """Response model for available indexes"""
    indexes: List[str] = Field(..., description="List of available index names")
