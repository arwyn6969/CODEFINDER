# CODEFINDER Demo Materials

This directory contains sample documents and demo scripts for showcasing CODEFINDER functionality.

## Structure

```
demo/
├── sample_documents/      # Sample PDFs and images for testing
├── scripts/               # Demo and test scripts
└── README.md              # This file
```

## Sample Documents

Place sample documents in `sample_documents/` directory. Recommended samples:

1. **Cipher Examples**: Documents with known cipher patterns
2. **Geometric Patterns**: Documents with geometric arrangements
3. **Cross-Document Samples**: Related documents for pattern matching
4. **Ancient Text Samples**: Historical documents for etymology analysis

## Running the Demo

### Quick Demo

```bash
# Run with default sample document
python scripts/demo.py

# Run with specific document
python scripts/demo.py --document path/to/document.pdf
```

### What the Demo Shows

1. **Document Upload**: Creates a document record in the database
2. **Processing Pipeline**: Runs through all analysis stages
3. **Pattern Detection**: Shows detected patterns and their confidence
4. **Results Summary**: Displays processing statistics and results

## Creating Sample Documents

If you don't have sample documents, you can:

1. **Generate Test PDFs**: Use any PDF generator to create test documents
2. **Use Public Domain Texts**: Download historical texts from Project Gutenberg
3. **Create Synthetic Data**: Generate documents with known patterns for testing

## Expected Results

After running the demo, you should see:

- Document processing status
- Number of pages processed
- Patterns detected (if any)
- Processing time
- Stage-by-stage results

## Troubleshooting

### No Sample Documents Found

Create the directory and add PDF files:
```bash
mkdir -p demo/sample_documents
# Add your PDF files here
```

### Database Errors

Ensure the database is initialized:
```bash
alembic upgrade head
```

### Tesseract Not Found

Install Tesseract OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Next Steps

After running the demo:

1. Explore the API endpoints using `/api/docs`
2. Try uploading different document types
3. Experiment with pattern detection
4. Generate reports for your documents
