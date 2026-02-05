# CODEFINDER User Guide

> **All-in-one OCR, analysis, and geometric/cipher exploration pipeline for historical texts**

---

## 📋 Table of Contents

1. [What is CODEFINDER?](#what-is-codefinder)
2. [Main Capabilities](#main-capabilities)
3. [Quick Start Guide](#quick-start-guide)
4. [Feature Deep Dive](#feature-deep-dive)
5. [How Features Work Together](#how-features-work-together)
6. [Running the Application](#running-the-application)
7. [Testing Guide](#testing-guide)
8. [Axiomatic Analysis Protocol](#axiomatic-analysis-protocol)
9. [API Reference](#api-reference)

---

## What is CODEFINDER?

CODEFINDER is a specialized research platform designed to analyze historical and ancient texts for hidden patterns. It combines:

- **Modern OCR** – Extract text from PDF and image documents
- **Cryptographic Analysis** – Detect and solve ciphers (Caesar, Atbash, Substitution)
- **Numerological Research** – Gematria calculation across English, Hebrew, and Greek
- **Bible Code Search** – Equidistant Letter Sequence (ELS) analysis
- **Geometric Pattern Detection** – Sacred geometry, mathematical constants
- **Cross-Document Correlation** – Find hidden connections between documents

### Who Is This For?

- Researchers studying historical ciphers and hidden messages
- Scholars analyzing biblical or ancient texts
- Anyone interested in pattern discovery in historical documents

---

## Main Capabilities

### 🔤 1. Document Processing & OCR

**What it does:** Upload PDF, images, or text files for automated text extraction.

**Key features:**
- Tesseract-based OCR with image preprocessing
- Automatic metadata extraction
- Background processing pipeline

### 🔢 2. Gematria Engine

**What it does:** Calculate numerical values of words using historical cipher systems.

**Supported ciphers:**
| Cipher | Description |
|--------|-------------|
| **Simple/Ordinal** | A=1, B=2... Z=26 |
| **Reverse** | Z=1, Y=2... A=26 |
| **Sumerian** | A=6, B=12... (×6) |
| **Bacon Simple** | 24-letter (I=J, U=V) |
| **Bacon Reverse** | Reversed Bacon-era alphabet |
| **Kay Cipher** | A=27 "Golden Key" method |
| **Hebrew Gematria** | Standard Mispar Ragil (Aleph-Tav) |
| **Greek Isopsephy** | Classical Greek values |

**Significant Numbers Auto-Detected:**
- **33** – Bacon (Simple)
- **67** – Francis (Simple)
- **100** – Francis Bacon (Simple)
- **157** – Fra Rosicrosse (Simple)
- **287** – Fra Rosicrosse (Kay)
- **888** – Jesus (Greek Isopsephy)
- **Pi, Phi, 666, 432** – Mathematical sacred numbers

### 🔍 3. ELS (Equidistant Letter Sequence) Analyzer

**What it does:** Search for hidden words formed by reading every Nth letter in a text.

**Key features:**
- Forward and backward directional scanning
- Configurable skip intervals (min/max)
- Built-in **Koren Torah** corpus (304,805 letters)
- 2D grid visualization of matches

### 🧩 4. Cipher Detection & Solving

**What it does:** Identify and decode encrypted text.

**Supported methods:**
- **Caesar Cipher** – Rotate letters by a key
- **Atbash Cipher** – Hebrew/English mirror substitution
- **Substitution Cipher** – Custom character mapping
- **Reverse** – Simple text reversal

### 📐 5. Geometric Analyzer

**What it does:** Detect mathematical patterns in text layouts and character positions.

**Detects:**
- Shapes: triangles, rectangles, circles, crosses
- Mathematical constants: π (Pi), φ (Golden Ratio)
- Harmonic intervals and Pythagorean relationships
- SVG path generation for visualization

### 🌍 6. BardCode Engine

**What it does:** Specialized "Sacred Geometry" detection based on Alan Green's research.

**Capabilities:**
- Vesica Piscis patterns
- Pythagorean triples (3:4:5)
- Pentagram structures
- Geographic coordinate mapping (Great Pyramid, Stonehenge, etc.)

### 🕸️ 7. Cross-Document Relationships

**What it does:** Find hidden connections between documents in your library.

**Analysis types:**
- Pairwise document similarity scores
- Shared pattern detection
- Stylistic authorship fingerprinting
- Network graph visualization

---

## Quick Start Guide

### Prerequisites

- Docker and Docker Compose
- Git

### 1. Clone and Start

```bash
cd /Users/arwynhughes/Documents/CODEFINDER_PUBLISH

# Start all services
docker-compose up -d
```

### 2. Access the Application

| Service | URL |
|---------|-----|
| **Frontend Dashboard** | http://localhost:3000 |
| **API Docs (Swagger)** | http://localhost:8000/docs |
| **Health Check** | http://localhost:8000/health |

### 3. First Steps

1. **Login** – Use demo credentials or register
2. **Upload a Document** – PDF, image, or text file
3. **View Analysis** – Automatic pattern detection runs in background
4. **Research Tools** – Use Gematria, ELS, or Cipher tools interactively

---

## Feature Deep Dive

### Three Operating Strategies

#### A. The "Dragnet" (Automatic Analysis)

When you upload a document, CODEFINDER automatically:
- Scans filename and first 100 characters (Incipit) for Gematria
- Runs background ELS skip-scan for common terms
- Saves significant patterns matching the "Interesting Numbers" registry

#### B. The "Microscope" (Interactive Research)

Use on-demand tools for specific analysis:
- **Gematria Tool** – Calculate any text across all ciphers
- **ELS Tool** – Search custom text or the Torah corpus
- **Cipher Solver** – Manually decode encrypted messages
- **Manual Pinning** – Save research findings to the library

#### C. The "Map" (Deep Connection Analysis)

Discover hidden bonds between documents:
- Documents sharing cryptographic signatures are linked
- Visualize relationships in the Network Graph
- Track evidence chains across your document collection

---

## How Features Work Together

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCUMENT UPLOAD                               │
│                    (PDF/Image/Text)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINE                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   OCR    │→ │  Store   │→ │ Metadata │→ │  Index   │        │
│  │ Extract  │  │   File   │  │ Extract  │  │  Create  │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   AUTOMATIC ANALYSIS                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Gematria │  │   ELS    │  │ Cipher   │  │ Anomaly  │        │
│  │  Scan    │  │  Scan    │  │ Detect   │  │ Detect   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              PATTERN DATABASE                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Saved Patterns: Gematria matches, ELS findings, Shapes   │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            CROSS-DOCUMENT ANALYSIS                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │ Shared   │  │ Author   │  │ Network  │                      │
│  │ Patterns │  │ Profiles │  │  Graph   │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

### Workflow Example: Analyzing a Historical Document

1. **Upload** – Add a PDF scan of Shakespeare's First Folio dedication
2. **Wait** – Pipeline extracts text, runs initial analysis
3. **Review** – Check the overview for significant patterns flagged
4. **Deep Dive** – Use Gematria to calculate "Francis Bacon" → 100 (Simple)
5. **ELS Search** – Scan for "BACON" hidden at skip intervals
6. **Geometry** – Check if letter positions form geometric patterns
7. **Connect** – See if patterns match other documents in your library

---

## Running the Application

### Using Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up -d --build
```

### Services Started

| Service | Port | Description |
|---------|------|-------------|
| `postgres` | 5432 | PostgreSQL database |
| `redis` | 6379 | Redis cache |
| `api` | 8000 | FastAPI backend |
| `frontend` | 3000 | React dashboard |

### Running API Only (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://analyzer:analyzer_pass@localhost:5432/ancient_text_analyzer"
export REDIS_URL="redis://localhost:6379"

# Run the API server
uvicorn app.api.main:app --reload --port 8000
```

### Running Frontend Only

```bash
cd frontend
npm install
npm run dev
```

---

## Testing Guide

CODEFINDER has extensive test coverage across all major components.

### Test Structure

```
tests/
├── test_api_endpoints.py          # All API route tests
├── test_anomaly_detector.py       # Anomaly detection service
├── test_bardcode_analyzer.py      # BardCode/Sacred geometry
├── test_cipher_detector.py        # Cipher identification
├── test_cipher_explanation_validator.py
├── test_cross_document_analyzer.py
├── test_cross_document_pattern_database.py
├── test_cross_reference_visualizer.py
├── test_database_models.py        # SQLAlchemy models
├── test_etymology_engine.py       # Word origin analysis
├── test_geometric_analyzer.py     # Shape detection
├── test_geometric_visualizer.py
├── test_grid_generator.py         # ELS grid generation
├── test_image_processor.py        # Image preprocessing
├── test_main.py                   # App initialization
├── test_ocr_engine.py             # Text extraction
├── test_pattern_significance_ranker.py
├── test_pdf_processor.py          # PDF handling
├── test_processing_pipeline.py    # Full pipeline
├── test_relationship_analyzer.py  # Cross-document
├── test_report_generator.py       # Report creation
├── test_sacred_geometry.py        # Geometric patterns
├── test_search_service.py         # Search functionality
├── test_text_analyzer.py          # Text analysis
└── test_text_grid_visualizer.py   # ELS visualization
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api_endpoints.py

# Run tests by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run with coverage
pytest --cov=app --cov-report=html
```

### Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Fast, isolated unit tests |
| `@pytest.mark.integration` | Tests requiring database/services |
| `@pytest.mark.slow` | Long-running tests |
| `@pytest.mark.asyncio` | Async test functions |

### Key Test Classes

| Test Class | Coverage |
|------------|----------|
| `TestAuthEndpoints` | Login, register, logout, refresh tokens |
| `TestDocumentEndpoints` | Upload, list, status checks |
| `TestAnalysisEndpoints` | Overview, analyze, geometric |
| `TestPatternEndpoints` | Pattern retrieval and detail |
| `TestVisualizationEndpoints` | Graph generation |
| `TestResearchEndpoints` | Gematria, ELS, Cipher |

### Example: Running ELS Tests

```bash
# Test the ELS analyzer service
pytest tests/test_grid_generator.py -v

# Test ELS API endpoints
pytest tests/test_api_endpoints.py -k "els" -v
```

---

## Axiomatic Analysis Protocol

CODEFINDER operates under a strict philosophical framework when analyzing historical texts, particularly the **1609 Shakespeare Sonnets**.

### The Axiom of Intent
> **"Every mark, ligature, spacing choice, and typographic anomaly is a meticulously placed data point with high geographical precision, intended for future decoding."**

### Operational Rules
1.  **No "Printer Errors"**: Anomalies are never dismissed as "typos" or "dirty proofs". They are classified as **High-Entropy Data Points**.
2.  **Geographical Precision**: The "Digital Type Case" is a coordinate map. The location (X, Y) of an anomaly is as significant as the character itself.
3.  **Variant Density**: Distinctions between character sorts (e.g., 'Type A e' vs 'Type B e') are treated as intentional encoding variations, not manufacturing defects.

This protocol overrides standard bibliographic assumptions within the CODEFINDER system.

---

## API Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
All endpoints (except `/auth` and `/health`) require JWT authentication.

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "demo", "password": "demo"}'

# Use token
curl http://localhost:8000/api/v1/documents \
  -H "Authorization: Bearer <your-token>"
```

### Key Endpoints

#### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload a file |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}` | Get document details |
| `GET` | `/documents/{id}/status` | Processing status |

#### Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analysis/{id}/analyze` | Run full analysis |
| `GET` | `/analysis/{id}/overview` | Pattern summary |
| `GET` | `/analysis/{id}/geometric` | Geometric patterns |
| `GET` | `/analysis/{id}/cipher` | Cipher detection |
| `GET` | `/analysis/{id}/anomalies` | Anomaly list |

#### Research
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/research/gematria` | Calculate Gematria |
| `POST` | `/research/els` | ELS search |
| `POST` | `/research/els/visualize` | Generate ELS grid |
| `POST` | `/research/cipher/solve` | Solve a cipher |

#### Relationships
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/relationships/network` | Generate network graph |

#### Visualizations
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/visualizations/{id}/geometric` | SVG paths for shapes |

### Example: Gematria Calculation

```bash
curl -X POST http://localhost:8000/api/v1/research/gematria \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Francis Bacon",
    "save": false
  }'
```

**Response:**
```json
{
  "text": "Francis Bacon",
  "values": {
    "simple": 100,
    "reverse": 182,
    "bacon_simple": 100,
    "bacon_kay": 397,
    "hebrew": null,
    "greek": null
  },
  "matches": ["100: Francis Bacon (Simple)"]
}
```

### Example: ELS Search in Torah

```bash
curl -X POST http://localhost:8000/api/v1/research/els \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "torah",
    "terms": ["TORAH", "MOSES"],
    "min_skip": 2,
    "max_skip": 100
  }'
```

---

## Appendix: Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                  │
│               React + D3 + Ant Design                           │
│                    Port 3000                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API LAYER                                   │
│              FastAPI + SQLAlchemy                               │
│                    Port 8000                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Routes: /auth /documents /analysis /research /patterns    │  │
│  │         /reports /search /relationships /visualizations   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICES                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Processing      │  │ Analysis        │  │ Visualization   │  │
│  │ ─────────────── │  │ ────────────────│  │ ────────────────│  │
│  │ • OCR Engine    │  │ • Gematria      │  │ • Grid Gen      │  │
│  │ • PDF Processor │  │ • ELS Analyzer  │  │ • ELS Visualizer│  │
│  │ • Image Proc    │  │ • Cipher Detect │  │ • Geo Visualizer│  │
│  │ • Pipeline      │  │ • Anomaly Det   │  │ • Cross-Ref Vis │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Geometry        │  │ Cross-Document  │  │ Research        │  │
│  │ ─────────────── │  │ ────────────────│  │ ────────────────│  │
│  │ • Geo Analyzer  │  │ • Relationship  │  │ • Etymology     │  │
│  │ • BardCode      │  │ • Pattern DB    │  │ • Significance  │  │
│  │ • Sacred Geo    │  │ • Correlation   │  │ • Text Analyzer │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
│  ┌──────────────────────────┐  ┌─────────────────────────────┐  │
│  │ PostgreSQL               │  │ Redis                       │  │
│  │ • Documents              │  │ • Session cache             │  │
│  │ • Patterns               │  │ • Rate limiting             │  │
│  │ • Pages                  │  │ • Background jobs           │  │
│  │ • Users                  │  │                             │  │
│  │ Port 5432                │  │ Port 6379                   │  │
│  └──────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Need Help?

- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **OpenAPI Spec**: `openapi.json` in project root
- **GitHub Issues**: Report bugs or request features
