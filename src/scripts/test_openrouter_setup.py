#!/usr/bin/env python3
"""
Test script to verify OpenRouter API connection and DeepSeek R1 model access.
"""

import json
import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


def test_openrouter_connection():
    """Test OpenRouter API connection with DeepSeek R1."""
    
    # Try both uppercase and lowercase variants
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenRouter API key")
        return False
    
    print("🔑 API key found")
    
    # Test API call
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/test",
        "X-Title": "DeepSeek Test"
    }
    
    payload = {
        "model": "deepseek/deepseek-r1",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France? Please use <think></think> tags for your reasoning."
            }
        ],
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    print("🚀 Testing API connection...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            
            print("✅ API connection successful!")
            print(f"📊 Tokens used - Input: {usage.get('prompt_tokens', 0)}, Output: {usage.get('completion_tokens', 0)}")
            print(f"💬 Sample response: {content[:200]}...")
            
            # Check if thinking tags are present
            if '<think>' in content and '</think>' in content:
                print("🧠 DeepSeek R1 reasoning detected!")
            else:
                print("⚠️  No thinking tags found in response")
            
            return True
            
        else:
            print(f"❌ API error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_sample_processing():
    """Test processing a sample from the dataset."""
    # File is in data/medmcqa/ directory relative to scripts
    sample_file = Path(__file__).parent.parent.parent / "data" / "medmcqa" / "train_reasoning_sample.json"
    
    if not sample_file.exists():
        print(f"❌ train_reasoning_sample.json not found at {sample_file}")
        return False
    
    print("📄 Loading sample data...")
    
    # Load first sample
    with open(sample_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
    
    print(f"🏥 Sample question: {sample['question'][:100]}...")
    print(f"📚 Subject: {sample['subject_name']}")
    print(f"✅ Correct answer: {sample['cop']} ({['A', 'B', 'C', 'D'][sample['cop']-1]})")
    
    return True

def main():
    """Main test function."""
    print("🧪 DeepSeek Reasoning Generator Test")
    print("=" * 40)
    
    # Test 1: Environment setup
    print("\n1️⃣ Testing environment setup...")
    if not test_sample_processing():
        return
    
    # Test 2: API connection
    print("\n2️⃣ Testing OpenRouter API connection...")
    if not test_openrouter_connection():
        return
    
    print("\n🎉 All tests passed! Ready to generate reasoning data.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements_reasoning.txt")
    print("2. Run the main script: python generate_deepseek_reasoning.py")

if __name__ == "__main__":
    main()
