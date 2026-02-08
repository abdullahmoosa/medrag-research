#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration management for RAG service
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class RAGConfig:
    """Configuration for RAG service"""
    
    # Index configuration
    index_name: str = "medcorp_medembed"
    index_dir: Optional[str] = None
    
    # Embedding configuration
    embed_model: str = "abhinand/MedEmbed-large-v0.1"
    device: str = "auto"  # auto, cpu, cuda
    
    # Retrieval configuration
    mode: str = "hybrid"  # hybrid, dense, bm25
    k: int = 12
    dense_k: int = 80
    bm25_k: int = 400
    rrf_k: int = 60
    
    # Evidence packing
    max_evidence_tokens: int = 1200
    hf_tokenizer: str = "deepseek-ai/DeepSeek-R1"
    
    # LLM configuration
    llm_model: str = "thewindmom/llama3-med42-8b"
    ollama_base_url: str = "http://localhost:11434"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    
    # System configuration
    batch_size: int = 1
    workers: int = 4
    
    def __post_init__(self):
        """Set default index directory if not provided"""
        if self.index_dir is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.index_dir = os.path.join(project_root, "indexes", self.index_name)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update index directory if index name changed
        if "index_name" in config_dict:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            self.index_dir = os.path.join(project_root, "indexes", self.index_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "index_name": self.index_name,
            "index_dir": self.index_dir,
            "embed_model": self.embed_model,
            "device": self.device,
            "mode": self.mode,
            "k": self.k,
            "dense_k": self.dense_k,
            "bm25_k": self.bm25_k,
            "rrf_k": self.rrf_k,
            "max_evidence_tokens": self.max_evidence_tokens,
            "hf_tokenizer": self.hf_tokenizer,
            "llm_model": self.llm_model,
            "ollama_base_url": self.ollama_base_url,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "batch_size": self.batch_size,
            "workers": self.workers,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """Create configuration from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not os.path.exists(self.index_dir):
            raise ValueError(f"Index directory does not exist: {self.index_dir}")
        
        if self.mode not in ["hybrid", "dense", "bm25"]:
            raise ValueError(f"Invalid retrieval mode: {self.mode}")
        
        if self.device not in ["auto", "cpu", "cuda"]:
            raise ValueError(f"Invalid device: {self.device}")
        
        if self.k <= 0 or self.dense_k <= 0 or self.bm25_k <= 0:
            raise ValueError("All k values must be positive")
        
        if self.max_evidence_tokens <= 0:
            raise ValueError("max_evidence_tokens must be positive")
        
        return True
