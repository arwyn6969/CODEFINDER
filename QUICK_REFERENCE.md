# CODEFINDER Quick Reference Guide

## Project Overview
**CODEFINDER** is an Ancient Text Analysis System that performs OCR, pattern detection, geometric analysis, cipher detection, and cross-document pattern matching on historical documents.

## Tech Stack Summary

### Backend
- **Framework**: FastAPI 0.100+
- **Language**: Python 3.11
- **Database**: PostgreSQL (prod), SQLite (dev)
- **ORM**: SQLAlchemy 2.0+
- **Migrations**: Alembic
- **OCR**: Tesseract (pytesseract)
- **Image Processing**: OpenCV, Pillow, PyMuPDF

### Frontend
- **Framework**: React
- **UI Library**: Ant Design
- **Routing**: React Router
- **Real-time**: WebSocket

### Infrastructure
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Caching**: Redis (configured but not used)
- **Background Jobs**: Celery (listed but not used)

## Project Structure

```
/workspace
├── app/                    # Backend application
│   ├── api/               # API routes and middleware
│   ├── core/              # Configuration and database
│   ├── models/            # Database models
│   └── services/          # Business logic (20+ services)
├── frontend/              # React frontend
├── tests/                 # Test suite (29 test files)
├── alembic/               # Database migrations
├── templates/             # Report templates
├── docker-compose.yml     # Local development setup
├── Dockerfile            # Container definition
└── requirements.txt       # Python dependencies
```

## Key Features

1. **Document Processing**
   - PDF/image upload
   - OCR extraction with confidence tracking
   - Multi-page document support

2. **Analysis Capabilities**
   - Pattern detection (cipher, geometric, linguistic)
   - Geometric measurements and sacred geometry
   - Etymology and linguistic analysis
   - Cross-document pattern matching
   - Grid-based text analysis

3. **Real-time Updates**
   - WebSocket progress tracking
   - Background processing

4. **Reporting**
   - HTML report generation
   - Pattern visualization
   - Analysis summaries

## Quick Start

### Prerequisites
- Python 3.11
- PostgreSQL (or SQLite for dev)
- Tesseract OCR
- Node.js (for frontend)

### Local Development
```bash
# Start services
docker-compose up -d

# Run migrations
alembic upgrade head

# Start backend
uvicorn app.api.main:app --reload

# Start frontend (from frontend/)
npm install
npm start
```

## Critical Issues to Address

### 🔴 Security (Critical)
- [ ] Remove hardcoded secret keys
- [ ] Implement proper user authentication
- [ ] Add input validation
- [ ] Secure CORS configuration

### 🟡 Configuration (High)
- [ ] Create `.env.example`
- [ ] Move secrets to environment variables
- [ ] Pin dependency versions
- [ ] Remove unused dependencies

### 🟢 Code Quality (Medium)
- [ ] Split large files (processing_pipeline.py)
- [ ] Standardize error handling
- [ ] Add type hints consistently
- [ ] Improve documentation

## Database Models

### Core Models
- **Document**: Main document entity
- **Page**: Individual pages with OCR results
- **Character**: Character-level OCR data
- **Word**: Word-level analysis
- **Pattern**: Detected patterns
- **Grid**: Text grid configurations

### Cross-Document Models
- **CrossDocumentPattern**: Patterns across documents
- **CrossPatternInstance**: Specific pattern instances
- **PatternRelationship**: Relationships between patterns

## API Endpoints

### Authentication
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout

### Documents
- `POST /api/documents/upload` - Upload document
- `GET /api/documents` - List documents
- `GET /api/documents/{id}` - Get document
- `GET /api/documents/{id}/status` - Processing status

### Analysis
- `POST /api/analysis/{document_id}` - Run analysis
- `GET /api/analysis/{document_id}/results` - Get results

### Patterns
- `GET /api/patterns` - List patterns
- `GET /api/patterns/{id}` - Get pattern details

### Search
- `POST /api/search` - Search documents/text

### Reports
- `GET /api/reports/{document_id}` - Generate report

### WebSocket
- `WS /api/ws/{document_id}` - Real-time updates

## Services Overview

### Core Services
- **ProcessingPipeline**: Main orchestration (1000+ lines)
- **OCR Engine**: Tesseract-based OCR
- **PDF Processor**: PDF extraction
- **Image Processor**: Image preprocessing

### Analysis Services
- **Text Analyzer**: Text analysis
- **Cipher Detector**: Cipher detection
- **Geometric Analyzer**: Geometric measurements
- **Etymology Engine**: Linguistic analysis
- **Anomaly Detector**: Anomaly detection
- **Pattern Significance Ranker**: Pattern ranking

### Cross-Document Services
- **Cross Document Analyzer**: Cross-document analysis
- **Cross Document Pattern Database**: Pattern database
- **Cross Reference Visualizer**: Visualization

### Visualization Services
- **Geometric Visualizer**: Geometric visualizations
- **Text Grid Visualizer**: Grid visualizations
- **Report Generator**: Report generation

## Testing

### Test Structure
- 29 test files
- Service tests
- API endpoint tests
- Model tests

### Running Tests
```bash
pytest                    # Run all tests
pytest -v                 # Verbose
pytest tests/test_*.py    # Specific test file
pytest --cov              # With coverage
```

## Configuration

### Environment Variables
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection (if used)
- `SECRET_KEY`: JWT secret key
- `DEBUG`: Debug mode (default: True)
- `ALLOWED_ORIGINS`: CORS origins

### Settings File
`app/core/config.py` - Pydantic Settings

## Deployment

### Docker
```bash
docker-compose up -d      # Development
docker build -t codefinder .  # Build image
```

### Production Considerations
- Use PostgreSQL (not SQLite)
- Set `DEBUG=False`
- Configure proper CORS
- Use environment variables for secrets
- Set up proper logging
- Configure health checks

## Common Tasks

### Add New Analysis Service
1. Create service in `app/services/`
2. Add to `ProcessingPipeline`
3. Create route in `app/api/routes/`
4. Add tests

### Add New Database Model
1. Create model in `app/models/`
2. Create Alembic migration: `alembic revision --autogenerate -m "description"`
3. Apply migration: `alembic upgrade head`
4. Update relationships if needed

### Add New API Endpoint
1. Add route in `app/api/routes/`
2. Add to router in `app/api/main.py`
3. Add tests
4. Update API documentation

## Known Issues

1. **Unused Dependencies**: Celery and Redis configured but not used
2. **Large Files**: `processing_pipeline.py` is 1000+ lines
3. **Security**: Hardcoded secrets, basic authentication
4. **Documentation**: Minimal documentation
5. **Frontend**: package.json needs verification

## Next Steps

1. **Immediate**: Address security issues
2. **Short-term**: Improve configuration and documentation
3. **Medium-term**: Refactor code and improve testing
4. **Long-term**: Implement scalability improvements

## Resources

- **Main Review**: `PROJECT_REVIEW.md`
- **Section Analysis**: `SECTION_BY_SECTION_ANALYSIS.md`
- **This Guide**: `QUICK_REFERENCE.md`

## Contact & Support

For questions or issues, refer to the main review documents for detailed recommendations and action plans.
