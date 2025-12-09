# CODEFINDER Quick Start Guide

Get up and running with CODEFINDER in 5 minutes!

## Prerequisites

- Python 3.11+
- PostgreSQL (or SQLite for development)
- Tesseract OCR
- Node.js 16+ (for frontend)

## 1. Clone and Setup (2 minutes)

```bash
# Clone repository
git clone <repository-url>
cd CODEFINDER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env and set SECRET_KEY (generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")
```

## 2. Initialize Database (1 minute)

```bash
# Run migrations
alembic upgrade head
```

## 3. Start Backend (1 minute)

```bash
# Start FastAPI server
uvicorn app.api.main:app --reload
```

Backend will be available at: http://localhost:8000
API docs at: http://localhost:8000/api/docs

## 4. Run Demo (1 minute)

```bash
# Run demo script (if you have sample documents)
python scripts/demo.py

# Or upload a document via API
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@path/to/document.pdf"
```

## 5. Start Frontend (Optional)

```bash
cd frontend
npm install
npm start
```

Frontend will be available at: http://localhost:3000

## Quick Test

Test the health endpoint:
```bash
curl http://localhost:8000/api/health
```

## Next Steps

1. **Explore API**: Visit http://localhost:8000/api/docs
2. **Run Tests**: `pytest`
3. **Check Coverage**: `pytest --cov=app --cov-report=html`
4. **Read Documentation**: See README.md and ARCHITECTURE.md

## Troubleshooting

### Database Connection Error
- Check DATABASE_URL in .env
- Ensure PostgreSQL is running (or use SQLite for dev)

### Tesseract Not Found
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS  
brew install tesseract
```

### Port Already in Use
Change port: `uvicorn app.api.main:app --port 8001`

## Docker Quick Start (Alternative)

```bash
# Start everything with Docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

That's it! You're ready to analyze documents! 🚀
