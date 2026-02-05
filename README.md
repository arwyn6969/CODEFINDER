# CODEFINDER 🔍

> **All-in-one OCR, analysis, and geometric/cipher exploration pipeline for historical texts**

[![CI](https://github.com/arwyn6969/CODEFINDER/actions/workflows/ci.yml/badge.svg)](https://github.com/arwyn6969/CODEFINDER/actions)

---

## 🌟 Overview

CODEFINDER is a specialized research platform for analyzing **historical and ancient texts** to discover hidden patterns, cryptographic encodings, and mathematical relationships. It combines modern OCR technology with advanced cipher detection, numerological analysis, and geometric pattern recognition.

### Key Features

| Feature | Description |
|---------|-------------|
| 🔤 **OCR Processing** | Extract text from PDF and image documents using Tesseract |
| 🔢 **Gematria Engine** | Calculate numerical values across 8+ cipher systems (Simple, Reverse, Sumerian, Bacon, Kay, Hebrew, Greek) |
| 🔍 **ELS Analyzer** | Equidistant Letter Sequence search with built-in Torah corpus |
| 🧩 **Cipher Detection** | Identify and solve Caesar, Atbash, and substitution ciphers |
| 📐 **Geometric Analysis** | Detect sacred geometry, mathematical constants (π, φ), and Pythagorean relationships |
| 🌍 **BardCode Engine** | Alan Green-style sacred geometry detection |
| 🕸️ **Cross-Document Analysis** | Find hidden connections between documents in your library |
| 🐸 **Prophetic Analysis** | Detect triple-term convergences in Torah (e.g. PEPE-MEME-FROG) with visualization |

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- Git

### 1. Clone & Start

```bash
git clone https://github.com/arwyn6969/CODEFINDER.git
cd CODEFINDER

# Start all services
docker-compose up -d
```

### 2. Access the Application

| Service | URL |
|---------|-----|
| **Frontend Dashboard** | http://localhost:3000 |
| **API Docs (Swagger)** | http://localhost:8000/api/docs |
| **Health Check** | http://localhost:8000/api/health |

### 3. First Steps

1. **Login** – Use demo credentials or register a new account
2. **Upload a Document** – PDF, image, or text file
3. **View Analysis** – Automatic pattern detection runs in background
4. **Research Tools** – Use Gematria, ELS, or Cipher tools interactively

---

## 📋 Documentation

- **[📘 User Guide](./CODEFINDER_USER_GUIDE.md)** – Comprehensive feature documentation
- **[🔬 Research Compendium](./docs/RESEARCH_COMPENDIUM.md)** – Consolidated research findings
- **[🔧 API Reference](http://localhost:8000/api/docs)** – Interactive Swagger UI

---

## 🏗️ Architecture

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
│              FastAPI + SQLAlchemy + Alembic                     │
│                    Port 8000                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SERVICES (26+)                                │
│  OCR • PDF • Image • Text • Grid • Geometry • Etymology         │
│  Gematria • ELS • Cipher • BardCode • Cross-Document            │
│  Anomaly Detection • Pattern Ranking • Visualization            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
│  PostgreSQL (Port 5432)    │    Redis (Port 6379)               │
│  Documents • Patterns      │    Session Cache • Jobs            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api_endpoints.py

# Run with coverage
pytest --cov=app --cov-report=html
```

**Test Coverage**: 600+ tests across API endpoints, services, and models.

---

## 🛠️ Development

### Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://analyzer:analyzer_pass@localhost:5432/ancient_text_analyzer"
export REDIS_URL="redis://localhost:6379"

# Run the API server
uvicorn app.api.main:app --reload --port 8000

# Run frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Tech Stack

- **Backend**: FastAPI + SQLAlchemy + Alembic
- **OCR**: Tesseract (via pytesseract)
- **Frontend**: React + D3.js + Ant Design
- **Database**: PostgreSQL
- **Cache**: Redis
- **CI/CD**: GitHub Actions

---

## 📂 Project Structure

```
CODEFINDER/
├── app/                    # Main application
│   ├── agents/             # Specialized OCR/analysis agents
│   ├── api/                # FastAPI routes and middleware
│   ├── core/               # Database and config
│   ├── models/             # SQLAlchemy models
│   ├── services/           # Business logic (26 services)
│   └── templates/          # Report templates
├── archive/                # Archived research scripts
├── docs/                   # Research documentation
├── frontend/               # React application
├── tests/                  # Pytest test suite (600+ tests)
├── alembic/                # Database migrations
└── docker-compose.yml      # Container orchestration
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Alan Green** – Inspiration for BardCode-style sacred geometry analysis
- **Tesseract OCR** – Open-source OCR engine
- **FastAPI** – Modern Python web framework

---

*Built with ❤️ for historical text researchers and cipher enthusiasts*
