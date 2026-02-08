#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for distributed Medical RAG architecture
Tests both backend API and frontend connectivity
"""

import requests
import time
import json
from typing import Dict, Any

class MedicalRAGTester:
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def test_backend_health(self) -> bool:
        """Test if backend is healthy and responding"""
        try:
            response = self.session.get(f"{self.backend_url}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Backend Health Check: {data['status']}")
                print(f"   Service: {data.get('service', 'Unknown')}")
                print(f"   Initialized: {data.get('initialized', False)}")
                if 'index' in data:
                    print(f"   Index: {data['index']}")
                    print(f"   LLM Model: {data['llm_model']}")
                    print(f"   Device: {data['device']}")
                return True
            else:
                print(f"❌ Backend Health Check Failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Backend Health Check Failed: {str(e)}")
            return False
    
    def test_chat_api(self) -> bool:
        """Test the chat API with a sample question"""
        test_message = "What are the main symptoms of diabetes?"
        
        try:
            print(f"\n📝 Testing Chat API with: '{test_message}'")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.backend_url}/api/chat",
                json={
                    "message": test_message,
                    "conversation_id": "test_session"
                },
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Chat API Response received in {elapsed:.2f}s")
                print(f"   Response length: {len(data['response'])} characters")
                print(f"   Sources found: {data['num_sources']}")
                print(f"   Retrieval time: {data.get('retrieval_time', 0):.2f}s")
                print(f"   Generation time: {data.get('generation_time', 0):.2f}s")
                
                # Show first 200 chars of response
                preview = data['response'][:200] + "..." if len(data['response']) > 200 else data['response']
                print(f"   Response preview: {preview}")
                
                return True
            else:
                print(f"❌ Chat API Failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Chat API Failed: {str(e)}")
            return False
    
    def test_config_api(self) -> bool:
        """Test configuration API"""
        try:
            # Get current config
            response = self.session.get(f"{self.backend_url}/api/config", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Config API - Get: Success")
                print(f"   Current mode: {data['config'].get('mode', 'Unknown')}")
                print(f"   Current k: {data['config'].get('k', 'Unknown')}")
                
                # Test config update (small change)
                update_response = self.session.post(
                    f"{self.backend_url}/api/config",
                    json={"temperature": 0.1},
                    timeout=10
                )
                
                if update_response.status_code == 200:
                    print(f"✅ Config API - Update: Success")
                    return True
                else:
                    print(f"❌ Config Update Failed: {update_response.status_code}")
                    return False
            else:
                print(f"❌ Config API Failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Config API Failed: {str(e)}")
            return False
    
    def test_available_indexes(self) -> bool:
        """Test available indexes endpoint"""
        try:
            response = self.session.get(f"{self.backend_url}/api/available-indexes", timeout=10)
            if response.status_code == 200:
                data = response.json()
                indexes = data.get('indexes', [])
                print(f"✅ Available Indexes API: {len(indexes)} indexes found")
                for idx in indexes:
                    if isinstance(idx, dict):
                        status = "✓" if idx.get('valid', False) else "✗"
                        print(f"   {status} {idx['name']}")
                    else:
                        print(f"   • {idx}")
                return True
            else:
                print(f"❌ Available Indexes API Failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Available Indexes API Failed: {str(e)}")
            return False
    
    def test_stats_api(self) -> bool:
        """Test service statistics endpoint"""
        try:
            response = self.session.get(f"{self.backend_url}/api/stats", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Stats API: Service running")
                if 'index_stats' in data:
                    stats = data['index_stats']
                    print(f"   Documents: {stats.get('total_documents', 'Unknown')}")
                    print(f"   Index type: {stats.get('index_type', 'Unknown')}")
                return True
            else:
                print(f"❌ Stats API Failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Stats API Failed: {str(e)}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all backend tests"""
        print(f"🧪 Testing Medical RAG Backend API")
        print(f"Backend URL: {self.backend_url}")
        print("=" * 50)
        
        tests = [
            ("Backend Health", self.test_backend_health),
            ("Available Indexes", self.test_available_indexes),
            ("Configuration API", self.test_config_api),
            ("Statistics API", self.test_stats_api),
            ("Chat API", self.test_chat_api),  # This one takes longer
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n🔍 Testing {test_name}...")
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {str(e)}")
                results.append(False)
        
        print("\n" + "=" * 50)
        print("📊 Test Summary:")
        
        passed = sum(results)
        total = len(results)
        
        for i, (test_name, _) in enumerate(tests):
            status = "✅ PASS" if results[i] else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! Backend is working correctly.")
            return True
        else:
            print("⚠️  Some tests failed. Check the backend setup.")
            return False

def test_frontend_server(frontend_url: str = "http://localhost:3000") -> bool:
    """Test if frontend server is accessible"""
    try:
        response = requests.get(frontend_url, timeout=10)
        if response.status_code == 200:
            print(f"✅ Frontend server accessible at {frontend_url}")
            print(f"   Content type: {response.headers.get('content-type', 'Unknown')}")
            print(f"   Response size: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ Frontend server returned {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Frontend server not accessible: {str(e)}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Medical RAG distributed architecture")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000",
                       help="Backend API URL")
    parser.add_argument("--frontend-url", type=str, default="http://localhost:3000", 
                       help="Frontend server URL")
    parser.add_argument("--skip-frontend", action="store_true",
                       help="Skip frontend server test")
    
    args = parser.parse_args()
    
    print("🏥 Medical RAG Architecture Test Suite")
    print("=" * 60)
    
    # Test backend
    tester = MedicalRAGTester(args.backend_url)
    backend_ok = tester.run_all_tests()
    
    # Test frontend
    frontend_ok = True
    if not args.skip_frontend:
        print(f"\n🌐 Testing Frontend Server")
        print("=" * 30)
        frontend_ok = test_frontend_server(args.frontend_url)
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 Final Results:")
    print(f"   Backend API: {'✅ OK' if backend_ok else '❌ ISSUES'}")
    if not args.skip_frontend:
        print(f"   Frontend Server: {'✅ OK' if frontend_ok else '❌ ISSUES'}")
    
    if backend_ok and frontend_ok:
        print("\n🎉 All systems operational!")
        print("   You can now use the Medical RAG chat application.")
        print(f"   Frontend: {args.frontend_url}")
        print(f"   Backend API: {args.backend_url}")
    else:
        print("\n⚠️  Some components have issues:")
        if not backend_ok:
            print("   - Check backend server and dependencies")
            print("   - Ensure Ollama is running")  
            print("   - Verify indexes are available")
        if not frontend_ok:
            print("   - Check frontend server is running")
    
    return 0 if (backend_ok and frontend_ok) else 1

if __name__ == "__main__":
    exit(main())
