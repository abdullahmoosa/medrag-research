#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG Service for medical question answering
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scripts.build_medcorp_index import HybridIndex, MedEmbedClient, OllamaClient
from src.app.config import RAGConfig
from src.app.models import ChatResponse, Source
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Medical RAG service for question answering"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.index: Optional[HybridIndex] = None
        self.embed_client: Optional[Any] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.executor = ThreadPoolExecutor(max_workers=config.workers)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the RAG service"""
        try:
            logger.info("Initializing RAG service...")
            
            # Validate configuration
            self.config.validate()
            
            # Load tokenizer for evidence packing
            logger.info(f"Loading tokenizer: {self.config.hf_tokenizer}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.hf_tokenizer, 
                use_fast=True
            )
            
            # Initialize index
            logger.info(f"Loading index from: {self.config.index_dir}")
            self.index = HybridIndex(index_dir=self.config.index_dir)
            
            # Load index in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self.index.load)
            
            # Initialize embedding client
            logger.info(f"Initializing embedding client: {self.config.embed_model}")
            if self.config.embed_model == "abhinand/MedEmbed-large-v0.1":
                self.embed_client = MedEmbedClient(
                    model_name=self.config.embed_model,
                    device=self.config.device
                )
            else:
                # Use Ollama for other embedding models
                self.embed_client = OllamaClient(
                    base_url=self.config.ollama_base_url,
                    model_doc=self.config.embed_model,
                    model_query=self.config.embed_model,
                )
            
            self._initialized = True
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise
    
    async def update_config(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration and reinitialize if necessary"""
        old_config = self.config.to_dict()
        self.config.update(config_dict)
        
        # Check if we need to reinitialize
        critical_changes = ["index_name", "index_dir", "embed_model", "device", "hf_tokenizer"]
        needs_reinit = any(old_config.get(key) != getattr(self.config, key) for key in critical_changes)
        
        if needs_reinit:
            logger.info("Critical configuration changed, reinitializing...")
            await self.initialize()
        else:
            logger.info("Configuration updated")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not self.tokenizer:
            return len(text.split())  # Fallback to word count
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _pack_passages(self, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pack passages to fit within token budget"""
        if self.config.max_evidence_tokens <= 0:
            return passages
        
        packed = []
        total_tokens = 0
        
        for passage in passages:
            text = passage.get("text", "")
            tokens = self._count_tokens(text)
            
            if total_tokens + tokens <= self.config.max_evidence_tokens:
                packed.append(passage)
                total_tokens += tokens
            else:
                break
        
        return packed
    
    def _format_evidence(self, passages: List[Dict[str, Any]]) -> str:
        """Format retrieved passages as evidence"""
        if not passages:
            return ""
        
        evidence_parts = []
        for i, passage in enumerate(passages, 1):
            title = passage.get("title", "").strip()
            source = passage.get("source", "").strip()
            text = passage.get("text", "").strip()
            
            # Create header
            header_parts = []
            if title:
                header_parts.append(f"Title: {title}")
            if source:
                header_parts.append(f"Source: {source}")
            
            header = " | ".join(header_parts) if header_parts else "Medical Literature"
            
            evidence_parts.append(f"[{i}] {header}\n{text}")
        
        return "\n\n".join(evidence_parts)
    
    def _build_prompt(self, query: str, evidence: str, conversation_context: str = "") -> str:
        """Build prompt for LLM with conversation context"""
        base_instruction = "You are a medical expert assistant. Answer the following question based on the provided medical evidence."
        
        if conversation_context:
            context_section = f"\nConversation History:\n{conversation_context}\n"
        else:
            context_section = ""
        
        if evidence:
            return f"""{base_instruction} 
If the evidence is insufficient, you may use your medical knowledge, but clearly indicate when you're doing so.
Be accurate, helpful, and concise. Always maintain a professional medical tone.
Consider the conversation history when relevant to provide contextual continuity.
{context_section}
Medical Evidence:
{evidence}

Question: {query}

Answer:"""
        else:
            return f"""{base_instruction.replace('based on the provided medical evidence', 'based on your knowledge')}
Be accurate, helpful, and concise. Always maintain a professional medical tone.
If you're uncertain about something, please indicate that clearly.
Consider the conversation history when relevant to provide contextual continuity.
{context_section}
Question: {query}

Answer:"""
    
    async def _retrieve_passages(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant passages for query"""
        if not self.index or not self.embed_client:
            raise RuntimeError("RAG service not initialized")
        
        loop = asyncio.get_event_loop()
        
        # Perform retrieval in thread pool
        passages = await loop.run_in_executor(
            self.executor,
            lambda: self.index.search(
                query,
                self.embed_client,
                k=self.config.k,
                mode=self.config.mode,
                rrf_k=self.config.rrf_k,
                dense_k=self.config.dense_k,
                bm25_k=self.config.bm25_k
            )
        )
        
        # Pack passages to fit token budget
        packed_passages = self._pack_passages(passages)
        
        return packed_passages
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using LLM"""
        try:
            import ollama
            
            # Use thread pool for Ollama call
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                self.executor,
                lambda: ollama.chat(
                    model=self.config.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "num_predict": self.config.max_new_tokens,
                    }
                )
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating a response. Please try again or rephrase your question."
    
    async def generate_response(self, query: str, conversation_id: str = "default", 
                              conversation_context: str = "") -> ChatResponse:
        """Generate response to user query using RAG"""
        if not self._initialized:
            raise RuntimeError("RAG service not initialized")
        
        start_time = time.time()
        
        try:
            # Retrieve relevant passages
            retrieval_start = time.time()
            passages = await self._retrieve_passages(query)
            retrieval_time = time.time() - retrieval_start
            
            # Format evidence
            evidence = self._format_evidence(passages)
            
            # Build prompt with conversation context
            prompt = self._build_prompt(query, evidence, conversation_context)
            
            # Generate response
            generation_start = time.time()
            response_text = await self._generate_response(prompt)
            generation_time = time.time() - generation_start
            
            # Convert passages to Source objects
            sources = [
                Source(
                    doc_id=p.get("doc_id", ""),
                    title=p.get("title", ""),
                    text=p.get("text", ""),
                    source=p.get("source", ""),
                    score=float(p.get("score", 0.0)),
                    url=p.get("url", "")
                ) for p in passages
            ]
            
            total_time = time.time() - start_time
            
            logger.info(f"Generated response for query in {total_time:.2f}s "
                       f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
            
            return ChatResponse(
                response=response_text,
                sources=sources,
                conversation_id=conversation_id,
                num_sources=len(sources),
                retrieval_time=retrieval_time,
                generation_time=generation_time
            )
            
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            return ChatResponse(
                response=f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                sources=[],
                conversation_id=conversation_id,
                num_sources=0,
                retrieval_time=0.0,
                generation_time=0.0
            )
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
