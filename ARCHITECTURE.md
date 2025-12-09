# CODEFINDER Architecture Documentation

## System Overview

CODEFINDER is a microservices-oriented monolithic application built with FastAPI (backend) and React (frontend). The system is designed to process, analyze, and discover patterns in ancient and historical documents.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (React)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Dashboard│  │  Upload  │  │ Analysis │  │  Search  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│       │              │              │              │        │
│       └──────────────┴──────────────┴──────────────┘        │
│                          │                                    │
│                    WebSocket / REST                           │
└──────────────────────────┼────────────────────────────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API Layer (Routes)                       │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │   │
│  │  │ Auth │ │ Docs │ │Analysis│ │Pattern│ │Search│      │   │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘      │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │            Service Layer                               │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │   OCR      │  │  Pattern   │  │  Geometric  │    │   │
│  │  │  Engine    │  │  Detector  │  │  Analyzer   │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘    │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐    │   │
│  │  │ Etymology  │  │   Cipher   │  │  Cross-Doc  │    │   │
│  │  │   Engine   │  │  Detector  │  │  Analyzer   │    │   │
│  │  └────────────┘  └────────────┘  └────────────┘    │   │
│  │                                                      │   │
│  │         Processing Pipeline (Orchestrator)           │   │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                    │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │            Data Access Layer                            │   │
│  │              SQLAlchemy ORM                            │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┼────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
┌───────▼────────┐                  ┌─────────▼────────┐
│   PostgreSQL   │                  │   Redis (Optional) │
│   Database     │                  │   Cache/Rate Limit │
└────────────────┘                  └───────────────────┘
```

## Component Architecture

### 1. Frontend Layer

**Technology**: React 18+ with Ant Design

**Components**:
- **Pages**: Dashboard, DocumentUpload, DocumentList, DocumentAnalysis, SearchPage
- **Services**: AuthService, WebSocketService
- **State Management**: Component-level state (consider adding Redux/Zustand for complex state)

**Communication**:
- REST API for CRUD operations
- WebSocket for real-time progress updates
- Axios for HTTP requests

### 2. API Layer

**Technology**: FastAPI

**Structure**:
```
app/api/
├── main.py              # Application entry point, route registration
├── middleware.py        # CORS, logging, error handling
├── dependencies.py      # Authentication, database sessions
└── routes/             # Endpoint handlers
    ├── auth.py         # Authentication endpoints
    ├── documents.py    # Document management
    ├── analysis.py     # Analysis operations
    ├── patterns.py     # Pattern queries
    ├── search.py       # Search functionality
    ├── reports.py      # Report generation
    ├── visualizations.py  # Visualization endpoints
    └── websocket.py    # WebSocket handlers
```

**Key Features**:
- JWT-based authentication
- Dependency injection for database sessions
- Custom exception handling
- Request/response logging
- Rate limiting (in-memory, can be upgraded to Redis)

### 3. Service Layer

**Technology**: Python services with dependency injection

**Core Services**:

1. **ProcessingPipeline** (`processing_pipeline.py`)
   - Orchestrates document processing stages
   - Manages progress tracking
   - Handles error recovery
   - **Note**: Large file (1000+ lines), candidate for refactoring

2. **OCR Engine** (`ocr_engine.py`)
   - Tesseract-based OCR
   - Confidence tracking
   - Uncertainty region detection

3. **Analysis Services**:
   - `text_analyzer.py` - Text analysis and statistics
   - `cipher_detector.py` - Cipher pattern detection
   - `geometric_analyzer.py` - Geometric measurements
   - `etymology_engine.py` - Linguistic analysis
   - `anomaly_detector.py` - Anomaly detection
   - `pattern_significance_ranker.py` - Pattern ranking

4. **Cross-Document Services**:
   - `cross_document_analyzer.py` - Cross-document pattern matching
   - `cross_document_pattern_database.py` - Pattern database management

5. **Visualization Services**:
   - `geometric_visualizer.py` - Geometric visualizations
   - `text_grid_visualizer.py` - Grid visualizations
   - `report_generator.py` - Report generation

**Service Communication**:
- Direct method calls (synchronous)
- Consider event-driven architecture for scalability

### 4. Data Layer

**Technology**: SQLAlchemy 2.0+ with Alembic migrations

**Database Models**:

#### Core Models
- **Document**: Main document entity with metadata
- **Page**: Individual pages with OCR results
- **Character**: Character-level OCR data with positions
- **Word**: Word-level analysis data
- **Pattern**: Detected patterns and anomalies
- **Grid**: Text grid configurations

#### Cross-Document Models
- **CrossDocumentPattern**: Patterns appearing across documents
- **CrossPatternInstance**: Specific pattern instances
- **PatternRelationship**: Relationships between patterns

#### Supporting Models
- **UncertainRegion**: Low-confidence OCR regions
- **GeometricMeasurement**: Geometric measurements
- **EtymologyCache**: Cached etymology data

**Database Design Principles**:
- Normalized schema with proper relationships
- Comprehensive indexing for performance
- JSON fields for flexible data storage
- Cascade deletes for data integrity

## Data Flow

### Document Processing Flow

```
1. Upload Document
   ↓
2. Validate File (extension, size, content)
   ↓
3. Save to Filesystem
   ↓
4. Create Document Record
   ↓
5. Start Background Processing
   ↓
6. Processing Pipeline:
   ├─ PDF Processing (extract pages)
   ├─ Image Processing (preprocessing)
   ├─ OCR Extraction (per page)
   ├─ Text Analysis
   ├─ Grid Generation
   ├─ Geometric Analysis
   ├─ Cipher Detection
   ├─ Etymology Analysis
   ├─ Anomaly Detection
   ├─ Pattern Ranking
   └─ Cross-Document Analysis
   ↓
7. Update Document Status
   ↓
8. Notify via WebSocket
   ↓
9. Results Available via API
```

### Request Flow

```
Client Request
   ↓
FastAPI Middleware (CORS, logging, error handling)
   ↓
Route Handler
   ↓
Dependency Injection (auth, database session)
   ↓
Service Layer
   ↓
Data Access Layer (SQLAlchemy)
   ↓
Database
   ↓
Response (JSON/WebSocket)
```

## Security Architecture

### Authentication & Authorization
- **JWT-based authentication** with configurable expiration
- **User model**: Currently simple (can be extended with database persistence)
- **Rate limiting**: In-memory (can be upgraded to Redis for distributed systems)

### Input Validation
- **File validation**: Extension, size, content (magic numbers)
- **Path traversal protection**: Filename sanitization
- **SQL injection protection**: SQLAlchemy ORM (parameterized queries)
- **XSS protection**: Input sanitization in templates

### Configuration Security
- **Environment variables**: All secrets in environment
- **Secret key validation**: Minimum length requirements
- **CORS restrictions**: Configurable allowed origins
- **Trusted hosts**: Domain validation

## Error Handling

### Exception Hierarchy

```
CodeFinderException (base)
├── ValidationError (400)
├── AuthenticationError (401)
├── AuthorizationError (403)
├── NotFoundError (404)
├── ConflictError (409)
├── ProcessingError (500)
│   └── OCRError
├── DatabaseError (500)
├── FileError (400)
└── ConfigurationError (500)
```

### Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": {
      "field": "additional context"
    }
  }
}
```

## Performance Considerations

### Current Architecture
- **Synchronous processing**: Blocks request until completion
- **Local file storage**: Files stored on filesystem
- **In-memory rate limiting**: Doesn't scale across instances
- **No caching**: All queries hit database

### Scalability Improvements (Recommended)
1. **Async Processing**: Use Celery or FastAPI BackgroundTasks properly
2. **Redis Caching**: Cache expensive queries and analysis results
3. **Cloud Storage**: Move files to S3/cloud storage
4. **Database Optimization**: Query optimization, read replicas
5. **CDN**: Serve static assets via CDN

## Deployment Architecture

### Development
- Docker Compose with PostgreSQL, Redis, and API
- Hot reload enabled
- SQLite option for quick setup

### Production (Recommended)
- **Application**: FastAPI with Gunicorn/Uvicorn workers
- **Database**: PostgreSQL with connection pooling
- **Cache**: Redis for caching and rate limiting
- **Storage**: Cloud storage (S3) for uploaded files
- **Reverse Proxy**: Nginx for SSL termination and load balancing
- **Monitoring**: Application monitoring (Sentry, DataDog)
- **Logging**: Centralized logging (ELK stack)

## Future Architecture Considerations

### Microservices Migration (Optional)
If scaling requires, consider splitting into:
- **Document Service**: Upload and storage
- **Processing Service**: OCR and analysis
- **Pattern Service**: Pattern detection and matching
- **Search Service**: Full-text search
- **API Gateway**: Request routing and authentication

### Event-Driven Architecture
- Use message queue (RabbitMQ/Kafka) for async processing
- Event sourcing for audit trail
- CQRS for read/write separation

## Technology Decisions

### Why FastAPI?
- Modern async/await support
- Automatic API documentation
- Type hints and validation
- High performance

### Why SQLAlchemy 2.0?
- Modern async support
- Type hints
- Better performance
- Active development

### Why React?
- Component-based architecture
- Large ecosystem
- Good performance
- Easy to maintain

## Development Workflow

1. **Feature Development**: Create branch, implement feature
2. **Testing**: Write tests, run test suite
3. **Database Changes**: Create Alembic migration
4. **Code Review**: Submit PR, review changes
5. **CI/CD**: Automated tests run on GitHub Actions
6. **Deployment**: Manual or automated deployment

## Monitoring & Observability

### Current State
- Basic logging to file
- Request/response logging
- Error logging with stack traces

### Recommended Additions
- **Structured Logging**: JSON format for log aggregation
- **Metrics**: Prometheus metrics endpoint
- **Tracing**: Distributed tracing (OpenTelemetry)
- **Error Tracking**: Sentry integration
- **Health Checks**: Detailed health check endpoint
- **Performance Monitoring**: APM tool integration

---

**Last Updated**: 2024  
**Maintainer**: Development Team
