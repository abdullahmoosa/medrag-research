# Medical RAG Application - Makefile
# Provides convenient commands for setup and deployment

.PHONY: help setup setup-venv setup-docker clean test run-backend run-frontend docker-up docker-down

# Default target
help:
	@echo "🏥 Medical RAG Application - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup       - Auto-detect and run best setup method"
	@echo "  make setup-venv  - Setup with virtual environment"
	@echo "  make setup-docker- Setup with Docker"
	@echo ""
	@echo "Run Commands:"
	@echo "  make run-backend - Start backend server"
	@echo "  make run-frontend- Start frontend server"
	@echo "  make run-both    - Start both backend and frontend"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-up   - Start all services with Docker"
	@echo "  make docker-down - Stop all Docker services"
	@echo "  make docker-logs - View Docker logs"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make test        - Test the setup"
	@echo "  make clean       - Clean up build artifacts"
	@echo "  make install-models - Download required Ollama models"

# Auto-detect setup method
setup:
	@echo "🔍 Detecting best setup method..."
	@if command -v docker >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then \
		echo "✅ Docker detected - using Docker setup"; \
		make setup-docker; \
	else \
		echo "🐍 Using Python virtual environment setup"; \
		make setup-venv; \
	fi

# Virtual environment setup
setup-venv:
	@echo "📦 Setting up virtual environment..."
	python setup_environment.py
	@echo "✅ Virtual environment setup complete!"
	@echo "   Use: make run-backend or make run-frontend"

# Docker setup  
setup-docker:
	@echo "🐳 Setting up with Docker..."
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env file from template..."; \
		cp .env.example .env; \
	fi
	docker-compose build
	@echo "✅ Docker setup complete!"
	@echo "   Use: make docker-up"

# Run backend (auto-detect method)
run-backend:
	@if [ -f run_backend_venv.bat ] || [ -f run_backend_venv.sh ]; then \
		echo "🖥️ Starting backend with virtual environment..."; \
		if [ -f run_backend_venv.sh ]; then \
			./run_backend_venv.sh; \
		else \
			./run_backend_venv.bat; \
		fi; \
	else \
		echo "🖥️ Starting backend directly..."; \
		python start_backend.py --host 0.0.0.0 --port 8000; \
	fi

# Run frontend (auto-detect method)
run-frontend:
	@if [ -f run_frontend_venv.bat ] || [ -f run_frontend_venv.sh ]; then \
		echo "🌐 Starting frontend with virtual environment..."; \
		if [ -f run_frontend_venv.sh ]; then \
			./run_frontend_venv.sh; \
		else \
			./run_frontend_venv.bat; \
		fi; \
	else \
		echo "🌐 Starting frontend directly..."; \
		python start_frontend.py --host 0.0.0.0 --port 3000; \
	fi

# Run both backend and frontend (requires two terminals)
run-both:
	@echo "🚀 Starting both backend and frontend..."
	@echo "   Note: This requires two terminal windows"
	@echo "   Backend: make run-backend"
	@echo "   Frontend: make run-frontend"

# Docker commands
docker-up:
	@echo "🐳 Starting Docker services..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "   Backend: http://localhost:8000"
	@echo "   Frontend: http://localhost:3000"
	@echo "   View logs: make docker-logs"

docker-down:
	@echo "🛑 Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped"

docker-logs:
	@echo "📋 Viewing Docker logs..."
	docker-compose logs -f

# Test the setup
test:
	@echo "🧪 Testing Medical RAG setup..."
	python test_distributed.py --backend-url http://localhost:8000 --frontend-url http://localhost:3000

# Install Ollama models
install-models:
	@echo "🤖 Downloading required Ollama models..."
	ollama pull thewindmom/llama3-med42-8b
	ollama pull deepseek-ai/deepseek-r1:8b
	@echo "✅ Models downloaded!"

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	@# Remove virtual environment
	@if [ -d venv ]; then \
		echo "   Removing virtual environment..."; \
		rm -rf venv; \
	fi
	@# Remove Docker containers and images
	@if command -v docker >/dev/null 2>&1; then \
		echo "   Cleaning Docker..."; \
		docker-compose down --rmi all --volumes --remove-orphans 2>/dev/null || true; \
	fi
	@# Remove Python cache
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Development helpers
dev-backend:
	@echo "🔧 Starting backend in development mode..."
	python start_backend.py --host 0.0.0.0 --port 8000 --reload --log-level debug

dev-frontend:
	@echo "🔧 Starting frontend in development mode..."
	python start_frontend.py --host 0.0.0.0 --port 3000 --no-browser

# Health checks
health:
	@echo "🏥 Checking service health..."
	@curl -s http://localhost:8000/api/health | python -m json.tool || echo "Backend not responding"
	@curl -s http://localhost:3000 >/dev/null && echo "Frontend: ✅ OK" || echo "Frontend: ❌ Not responding"

# Show status
status:
	@echo "📊 Service Status:"
	@echo "=================="
	@echo -n "Backend (port 8000): "
	@curl -s http://localhost:8000/api/health >/dev/null && echo "✅ Running" || echo "❌ Stopped"
	@echo -n "Frontend (port 3000): "
	@curl -s http://localhost:3000 >/dev/null && echo "✅ Running" || echo "❌ Stopped"
	@echo -n "Ollama (port 11434): "
	@curl -s http://localhost:11434/api/tags >/dev/null && echo "✅ Running" || echo "❌ Stopped"

# Show URLs
urls:
	@echo "🔗 Service URLs:"
	@echo "==============="
	@echo "Backend API:    http://localhost:8000"
	@echo "Frontend UI:    http://localhost:3000"  
	@echo "API Docs:       http://localhost:8000/docs"
	@echo "Ollama:         http://localhost:11434"
