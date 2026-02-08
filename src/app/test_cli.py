#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI interface for testing the RAG service
"""

import os
import sys
import asyncio
import argparse
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.rag_service import RAGService
from src.app.config import RAGConfig

async def test_rag_service(config_overrides: Dict[str, Any] = None):
    """Test the RAG service with a sample query"""
    
    # Create configuration
    config = RAGConfig()
    if config_overrides:
        config.update(config_overrides)
    
    print(f"Testing RAG service with configuration:")
    print(f"  Index: {config.index_name}")
    print(f"  Mode: {config.mode}")
    print(f"  Embedding Model: {config.embed_model}")
    print(f"  LLM Model: {config.llm_model}")
    print(f"  Device: {config.device}")
    print()
    
    # Initialize service
    print("Initializing RAG service...")
    service = RAGService(config)
    
    try:
        await service.initialize()
        print("✅ RAG service initialized successfully!\n")
        
        # Test queries
        test_queries = [
            "What are the symptoms of diabetes?",
            "How is hypertension diagnosed?",
            "What are the treatment options for asthma?",
            "Explain the mechanism of action of ACE inhibitors",
            "What are the risk factors for cardiovascular disease?"
        ]
        
        while True:
            print("\nSelect a test query or enter your own:")
            for i, query in enumerate(test_queries, 1):
                print(f"  {i}. {query}")
            print(f"  {len(test_queries) + 1}. Enter custom query")
            print(f"  0. Exit")
            
            try:
                choice = input("\nYour choice: ").strip()
                
                if choice == "0":
                    break
                
                elif choice.isdigit() and 1 <= int(choice) <= len(test_queries):
                    query = test_queries[int(choice) - 1]
                
                elif choice == str(len(test_queries) + 1):
                    query = input("Enter your question: ").strip()
                    if not query:
                        continue
                
                else:
                    query = choice  # Treat as direct query
                    if not query:
                        continue
                
                print(f"\n🔍 Query: {query}")
                print("=" * 50)
                
                # Generate response
                response = await service.generate_response(query)
                
                print(f"\n🤖 Response:\n{response.response}")
                print(f"\n📊 Stats:")
                print(f"  - Sources found: {response.num_sources}")
                print(f"  - Retrieval time: {response.retrieval_time:.2f}s")
                print(f"  - Generation time: {response.generation_time:.2f}s")
                
                if response.sources:
                    print(f"\n📚 Sources:")
                    for i, source in enumerate(response.sources, 1):
                        print(f"  [{i}] {source.title or 'Unknown Title'}")
                        print(f"      Source: {source.source or 'Unknown'}")
                        print(f"      Score: {source.score:.3f}")
                        print(f"      Text: {source.text[:100]}...")
                        print()
                
                print("=" * 50)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Failed to initialize RAG service: {str(e)}")
        return

def main():
    parser = argparse.ArgumentParser(description="Test RAG service CLI")
    parser.add_argument("--index", type=str, default="medcorp_medembed",
                       help="Index name to use")
    parser.add_argument("--mode", type=str, default="hybrid",
                       choices=["hybrid", "dense", "bm25"],
                       help="Retrieval mode")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device for embedding model")
    parser.add_argument("--k", type=int, default=12,
                       help="Number of passages to retrieve")
    parser.add_argument("--llm-model", type=str, default="thewindmom/llama3-med42-8b",
                       help="LLM model to use")
    parser.add_argument("--embed-model", type=str, default="abhinand/MedEmbed-large-v0.1",
                       help="Embedding model to use")
    
    args = parser.parse_args()
    
    # Build config overrides
    config_overrides = {
        "index_name": args.index,
        "mode": args.mode,
        "device": args.device,
        "k": args.k,
        "llm_model": args.llm_model,
        "embed_model": args.embed_model,
    }
    
    # Remove None values
    config_overrides = {k: v for k, v in config_overrides.items() if v is not None}
    
    print("🏥 Medical RAG Service CLI Test")
    print("=" * 40)
    
    asyncio.run(test_rag_service(config_overrides))

if __name__ == "__main__":
    main()
