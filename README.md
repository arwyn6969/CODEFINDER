# CODEFINDER

**Ancient Text Analysis System** - A comprehensive platform for OCR, pattern detection, geometric analysis, cipher detection, and cross-document pattern matching on historical documents.

[![CI](https://github.com/arwyn6969/CODEFINDER/actions/workflows/ci.yml/badge.svg)](https://github.com/arwyn6969/CODEFINDER/actions/workflows/ci.yml)

## Overview

CODEFINDER is a sophisticated text analysis system designed to analyze ancient and historical documents. It combines OCR capabilities with advanced pattern detection, geometric analysis, etymology research, and cross-document pattern matching to help researchers discover hidden patterns and relationships in historical texts.

### Key Features

- **OCR Processing**: Tesseract-based OCR with confidence tracking and uncertainty region detection
- **Pattern Detection**: Multiple pattern types including ciphers, geometric patterns, linguistic patterns, and structural patterns
- **Geometric Analysis**: Sacred geometry detection, angle/distance measurements, and spatial relationship analysis
- **Cross-Document Analysis**: Pattern matching across multiple documents to find shared constructions
- **Etymology Engine**: Linguistic analysis and word origin research
- **Grid Analysis**: Text grid generation and pattern detection within grids
- **Real-time Processing**: WebSocket support for progress tracking during document processing
- **Comprehensive Reporting**: HTML/PDF report generation with visualizations

## Technology Stack

### Backend
- **Framework**: FastAPI 0.100+ (Python 3.11)
- **Database**: PostgreSQL (production), SQLite (development)
- **ORM**: SQLAlchemy 2.0+
- **Migrations**: Alembic
- **OCR**: Tesseract (via pytesseract)
- **Image Processing**: OpenCV, Pillow, PyMuPDF
- **Scientific Computing**: NumPy, SciPy, scikit-learn, NetworkX

### Frontend
- **Framework**: React 18+
- **UI Library**: Ant Design 5+
- **Routing**: React Router 6+
- **Real-time**: WebSocket

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Caching**: Redis (optional)

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ (or SQLite for development)
- Tesseract OCR
- Node.js 16+ and npm (for frontend)
- Docker and Docker Compose (optional, for containerized setup)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CODEFINDER
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and set required variables (see Configuration section)
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up database**
   ```bash
   # Run migrations
   alembic upgrade head
   ```

5. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

### Running the Application

#### Development Mode

**Option 1: Docker Compose (Recommended)**
```bash
docker-compose up -d
```

This starts:
- PostgreSQL database (port 5432)
- Redis (port 6379)
- FastAPI backend (port 8000)

Then start the frontend:
```bash
cd frontend
npm start
```

**Option 2: Manual Setup**

1. **Start database** (if using PostgreSQL)
   ```bash
   # PostgreSQL should be running
   # Or use SQLite (default for development)
   ```

2. **Start backend**
   ```bash
   uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Start frontend**
   ```bash
   cd frontend
   npm start
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/api/docs

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure the following:

#### Required (Production)
- `SECRET_KEY`: Secret key for JWT tokens (generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`)
- `DATABASE_URL`: Database connection string (PostgreSQL recommended for production)

#### Optional
- `DEBUG`: Set to `False` in production
- `PRODUCTION`: Set to `true` in production environments
- `ALLOWED_ORIGINS`: Comma-separated list of allowed CORS origins
- `REDIS_URL`: Redis connection URL (optional, for caching)
- `TESSERACT_CMD`: Path to tesseract executable (auto-detected if not set)

See `.env.example` for all available configuration options.

### Security Notes

⚠️ **Important Security Considerations:**

1. **Never commit `.env` file** - It contains sensitive information
2. **Change default SECRET_KEY** - Generate a secure key for production
3. **Use PostgreSQL in production** - SQLite is only for development
4. **Set DEBUG=False in production** - Prevents exposing internal errors
5. **Configure ALLOWED_ORIGINS** - Restrict CORS to your frontend domain
6. **Set TRUSTED_HOSTS** - Add your production domain names

## Project Structure

```
CODEFINDER/
├── app/                    # Backend application
│   ├── api/               # API routes and middleware
│   │   ├── routes/       # Endpoint handlers
│   │   ├── dependencies.py
│   │   └── middleware.py
│   ├── core/              # Core configuration
│   │   ├── config.py     # Settings management
│   │   ├── database.py   # Database setup
│   │   └── exceptions.py # Custom exceptions
│   ├── models/           # Database models
│   │   ├── database_models.py
│   │   └── cross_document_models.py
│   └── services/         # Business logic (20+ services)
│       ├── ocr_engine.py
│       ├── processing_pipeline.py
│       └── [analysis services]
├── frontend/              # React frontend
│   ├── src/
│   │   ├── pages/        # React pages
│   │   └── services/     # API services
│   └── package.json
├── tests/                 # Test suite
├── alembic/               # Database migrations
├── templates/             # Report templates
├── docker-compose.yml     # Local development setup
├── Dockerfile            # Container definition
└── requirements.txt      # Python dependencies
```

## API Documentation

Once the application is running, API documentation is available at:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### Key Endpoints

- `POST /api/auth/login` - User authentication
- `POST /api/documents/upload` - Upload document for analysis
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Get document details
- `GET /api/analysis/{document_id}/results` - Get analysis results
- `GET /api/patterns` - List detected patterns
- `POST /api/search` - Search documents and text
- `GET /api/reports/{document_id}` - Generate report
- `WS /api/ws/{document_id}` - WebSocket for real-time updates

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_ocr_engine.py

# Run with verbose output
pytest -v
```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "Description of changes"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

### Code Quality

The project follows Python best practices:
- Type hints where applicable
- Comprehensive docstrings
- Custom exception hierarchy
- Structured logging

### Adding New Features

1. **New Analysis Service**: Create service in `app/services/`, add to `ProcessingPipeline`
2. **New API Endpoint**: Add route in `app/api/routes/`, register in `app/api/main.py`
3. **New Database Model**: Create model in `app/models/`, create Alembic migration

## Deployment

### Production Checklist

- [ ] Set `DEBUG=False` in environment
- [ ] Set `PRODUCTION=true` in environment
- [ ] Configure PostgreSQL database
- [ ] Set secure `SECRET_KEY`
- [ ] Configure `ALLOWED_ORIGINS` for CORS
- [ ] Set `TRUSTED_HOSTS` for your domain
- [ ] Configure proper logging
- [ ] Set up SSL/TLS certificates
- [ ] Configure backup strategy
- [ ] Set up monitoring and alerting

### Docker Production

```bash
# Build production image
docker build -t codefinder:latest .

# Run with production settings
docker run -d \
  -e DEBUG=False \
  -e PRODUCTION=true \
  -e SECRET_KEY=your-secret-key \
  -e DATABASE_URL=postgresql://... \
  -p 8000:8000 \
  codefinder:latest
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Add your license information here]

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Acknowledgments

- Tesseract OCR team
- FastAPI community
- All contributors and researchers using this tool

---

**Note**: This project is actively maintained. For the latest updates and documentation, see the project repository.
