#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI Chat Application with RAG capabilities for Health Domain
"""

import os
import sys
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.app.rag_service import RAGService
from src.app.config import RAGConfig
from src.app.models import ChatRequest, ChatResponse, ConfigRequest

app = FastAPI(
    title="Medical RAG Chat API",
    description="A FastAPI application for medical question answering using RAG",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG service
rag_service: Optional[RAGService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG service on startup"""
    global rag_service
    config = RAGConfig()
    rag_service = RAGService(config)
    await rag_service.initialize()

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the chat interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical RAG Chat</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f5f5;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .container {
                display: flex;
                flex: 1;
                max-width: 1400px;
                margin: 0 auto;
                width: 100%;
            }
            
            .sidebar {
                width: 300px;
                background: white;
                border-right: 1px solid #e0e0e0;
                padding: 1rem;
                overflow-y: auto;
            }
            
            .chat-container {
                flex: 1;
                display: flex;
                flex-direction: column;
                background: white;
                margin: 1rem;
                border-radius: 10px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .chat-messages {
                flex: 1;
                padding: 1rem;
                overflow-y: auto;
                max-height: calc(100vh - 200px);
            }
            
            .message {
                margin-bottom: 1rem;
                padding: 0.75rem;
                border-radius: 10px;
                max-width: 80%;
            }
            
            .user-message {
                background: #667eea;
                color: white;
                margin-left: auto;
                text-align: right;
            }
            
            .bot-message {
                background: #f0f0f0;
                color: #333;
            }
            
            .sources {
                margin-top: 0.5rem;
                padding-top: 0.5rem;
                border-top: 1px solid #ddd;
                font-size: 0.85rem;
                color: #666;
            }
            
            .source-item {
                background: #e8f4fd;
                margin: 0.25rem 0;
                padding: 0.5rem;
                border-radius: 5px;
                border-left: 3px solid #667eea;
            }
            
            .input-container {
                padding: 1rem;
                border-top: 1px solid #e0e0e0;
                background: #fafafa;
                border-radius: 0 0 10px 10px;
            }
            
            .input-group {
                display: flex;
                gap: 0.5rem;
            }
            
            input[type="text"] {
                flex: 1;
                padding: 0.75rem;
                border: 1px solid #ddd;
                border-radius: 25px;
                font-size: 1rem;
                outline: none;
                transition: border-color 0.3s;
            }
            
            input[type="text"]:focus {
                border-color: #667eea;
            }
            
            button {
                padding: 0.75rem 1.5rem;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 25px;
                cursor: pointer;
                font-size: 1rem;
                transition: background 0.3s;
            }
            
            button:hover {
                background: #5a67d8;
            }
            
            button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .config-section {
                margin-bottom: 1rem;
            }
            
            .config-section h3 {
                margin-bottom: 0.5rem;
                color: #333;
                font-size: 1rem;
            }
            
            .config-section select,
            .config-section input {
                width: 100%;
                padding: 0.5rem;
                margin-bottom: 0.5rem;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 0.9rem;
            }
            
            .loading {
                display: none;
                text-align: center;
                padding: 1rem;
                color: #666;
            }
            
            .spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🏥 Medical RAG Chat Assistant</h1>
            <p>Ask medical questions and get evidence-based answers</p>
        </div>
        
        <div class="container">
            <div class="sidebar">
                <div class="config-section">
                    <h3>📊 Index Selection</h3>
                    <select id="indexSelect">
                        <option value="medcorp_medembed">MedCorp MedEmbed</option>
                        <option value="medembed">MedEmbed</option>
                        <option value="testbooks_medembed_flat">Textbooks MedEmbed</option>
                    </select>
                </div>
                
                <div class="config-section">
                    <h3>🔍 Retrieval Mode</h3>
                    <select id="modeSelect">
                        <option value="hybrid">Hybrid (FAISS + BM25)</option>
                        <option value="dense">Dense (FAISS only)</option>
                        <option value="bm25">BM25 only</option>
                    </select>
                </div>
                
                <div class="config-section">
                    <h3>⚙️ Parameters</h3>
                    <label>Top-K Results:</label>
                    <input type="number" id="topK" value="12" min="1" max="50">
                    
                    <label>Dense K:</label>
                    <input type="number" id="denseK" value="80" min="10" max="200">
                    
                    <label>BM25 K:</label>
                    <input type="number" id="bm25K" value="400" min="50" max="1000">
                    
                    <label>Max Evidence Tokens:</label>
                    <input type="number" id="maxTokens" value="1200" min="200" max="3000">
                </div>
                
                <div class="config-section">
                    <h3>🤖 LLM Model</h3>
                    <select id="llmModel">
                        <option value="thewindmom/llama3-med42-8b">Llama3-Med42-8B</option>
                        <option value="deepseek-ai/deepseek-r1:8b">DeepSeek-R1-8B</option>
                        <option value="meditron-7b">Meditron-7B</option>
                    </select>
                </div>
                
                <button onclick="updateConfig()">Update Configuration</button>
            </div>
            
            <div class="chat-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot-message">
                        <strong>Medical Assistant:</strong> Hello! I'm your medical RAG assistant. Ask me any health-related questions, and I'll provide evidence-based answers with sources. How can I help you today?
                    </div>
                </div>
                
                <div class="loading" id="loadingIndicator">
                    <div class="spinner"></div>
                    <span>Searching medical literature...</span>
                </div>
                
                <div class="input-container">
                    <div class="input-group">
                        <input type="text" id="messageInput" placeholder="Ask a medical question..." onkeypress="handleKeyPress(event)">
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;

                const chatMessages = document.getElementById('chatMessages');
                const loadingIndicator = document.getElementById('loadingIndicator');
                
                // Add user message
                const userDiv = document.createElement('div');
                userDiv.className = 'message user-message';
                userDiv.innerHTML = `<strong>You:</strong> ${message}`;
                chatMessages.appendChild(userDiv);
                
                // Clear input and scroll
                input.value = '';
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Show loading
                loadingIndicator.style.display = 'block';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            message: message,
                            conversation_id: 'default'
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Add bot response
                    const botDiv = document.createElement('div');
                    botDiv.className = 'message bot-message';
                    
                    let sourcesHtml = '';
                    if (data.sources && data.sources.length > 0) {
                        sourcesHtml = '<div class="sources"><strong>📚 Sources:</strong>';
                        data.sources.forEach((source, index) => {
                            sourcesHtml += `
                                <div class="source-item">
                                    <strong>[${index + 1}]</strong> ${source.title || 'Medical Literature'}<br>
                                    <small>${source.source || 'Unknown'} | Score: ${(source.score || 0).toFixed(3)}</small><br>
                                    <em>${(source.text || '').substring(0, 150)}...</em>
                                </div>
                            `;
                        });
                        sourcesHtml += '</div>';
                    }
                    
                    botDiv.innerHTML = `
                        <strong>Medical Assistant:</strong> ${data.response}
                        ${sourcesHtml}
                    `;
                    chatMessages.appendChild(botDiv);
                    
                } catch (error) {
                    const errorDiv = document.createElement('div');
                    errorDiv.className = 'message bot-message';
                    errorDiv.innerHTML = `<strong>Error:</strong> Failed to get response. Please try again.`;
                    chatMessages.appendChild(errorDiv);
                }
                
                // Hide loading and scroll
                loadingIndicator.style.display = 'none';
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            async function updateConfig() {
                const config = {
                    index_name: document.getElementById('indexSelect').value,
                    mode: document.getElementById('modeSelect').value,
                    k: parseInt(document.getElementById('topK').value),
                    dense_k: parseInt(document.getElementById('denseK').value),
                    bm25_k: parseInt(document.getElementById('bm25K').value),
                    max_evidence_tokens: parseInt(document.getElementById('maxTokens').value),
                    llm_model: document.getElementById('llmModel').value
                };
                
                try {
                    const response = await fetch('/config', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(config)
                    });
                    
                    if (response.ok) {
                        alert('Configuration updated successfully!');
                    } else {
                        alert('Failed to update configuration');
                    }
                } catch (error) {
                    alert('Error updating configuration');
                }
            }
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with RAG"""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        response = await rag_service.generate_response(
            query=request.message,
            conversation_id=request.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.post("/config")
async def update_config(request: ConfigRequest):
    """Update RAG configuration"""
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        await rag_service.update_config(request.dict())
        return {"status": "success", "message": "Configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating configuration: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "medical-rag-chat"}

@app.get("/available-indexes")
async def get_available_indexes():
    """Get list of available indexes"""
    indexes_dir = os.path.join(PROJECT_ROOT, "indexes")
    available_indexes = []
    
    if os.path.exists(indexes_dir):
        for item in os.listdir(indexes_dir):
            index_path = os.path.join(indexes_dir, item)
            if os.path.isdir(index_path):
                available_indexes.append(item)
    
    return {"indexes": available_indexes}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
