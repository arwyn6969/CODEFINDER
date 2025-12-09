#!/usr/bin/env python3
"""
CODEFINDER Demo Script

This script demonstrates the full workflow of the CODEFINDER system:
1. Upload a sample document
2. Process it through the analysis pipeline
3. Display detected patterns and results
4. Generate a report

Usage:
    python scripts/demo.py [--document path/to/document.pdf]
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import get_db, init_db
from app.models.database_models import Document
from app.services.processing_pipeline import ProcessingPipeline


async def demo_document_processing(document_path: Optional[str] = None):
    """
    Demonstrate the complete document processing workflow.
    
    Args:
        document_path: Path to document to process. If None, uses sample.
    """
    print("=" * 70)
    print("CODEFINDER Demo - Document Processing Workflow")
    print("=" * 70)
    print()
    
    # Initialize database
    print("📊 Initializing database...")
    try:
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️  Database init warning: {e}")
    print()
    
    # Get document path
    if not document_path:
        # Look for sample documents
        sample_dir = Path(__file__).parent.parent / "demo" / "sample_documents"
        if sample_dir.exists():
            pdf_files = list(sample_dir.glob("*.pdf"))
            if pdf_files:
                document_path = str(pdf_files[0])
                print(f"📄 Using sample document: {document_path}")
            else:
                print("❌ No sample documents found. Please provide a document path.")
                print(f"   Expected location: {sample_dir}")
                return
        else:
            print("❌ Sample documents directory not found.")
            print(f"   Create: {sample_dir}")
            print("   Or provide document path with --document flag")
            return
    else:
        if not Path(document_path).exists():
            print(f"❌ Document not found: {document_path}")
            return
        print(f"📄 Processing document: {document_path}")
    
    print()
    
    # Get database session
    db = next(get_db())
    
    try:
        # Step 1: Create document record
        print("📝 Step 1: Creating document record...")
        document = Document(
            filename=Path(document_path).name,
            original_filename=Path(document_path).name,
            file_path=document_path,
            file_size=Path(document_path).stat().st_size,
            upload_date=datetime.utcnow(),
            processing_status="uploaded",
            uploaded_by="demo_user"
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        print(f"✅ Document created with ID: {document.id}")
        print()
        
        # Step 2: Initialize processing pipeline
        print("⚙️  Step 2: Initializing processing pipeline...")
        pipeline = ProcessingPipeline(db)
        print("✅ Pipeline initialized")
        print()
        
        # Step 3: Process document
        print("🔄 Step 3: Processing document...")
        print("   This may take a few minutes depending on document size...")
        print()
        
        result = await pipeline.process_document_async(document_path, document.id)
        
        if result.success:
            print("✅ Processing completed successfully!")
            print()
            
            # Step 4: Display results
            print("📊 Step 4: Processing Results")
            print("-" * 70)
            print(f"Total Pages: {result.total_pages}")
            print(f"Processing Time: {result.total_duration:.2f} seconds")
            print(f"Stages Completed: {len([s for s in result.stage_results if s.status.value == 'completed'])}")
            print()
            
            # Display patterns found
            patterns = db.query(Pattern).filter(Pattern.document_id == document.id).all()
            if patterns:
                print(f"🔍 Patterns Detected: {len(patterns)}")
                for i, pattern in enumerate(patterns[:5], 1):  # Show first 5
                    print(f"   {i}. {pattern.pattern_type}: {pattern.pattern_name}")
                    print(f"      Confidence: {pattern.confidence:.2f}")
                    print(f"      Significance: {pattern.significance_score:.2f}" if pattern.significance_score else "")
                if len(patterns) > 5:
                    print(f"   ... and {len(patterns) - 5} more patterns")
                print()
            
            # Display stage results
            print("📋 Stage Results:")
            for stage_result in result.stage_results:
                status_icon = "✅" if stage_result.status.value == "completed" else "❌" if stage_result.status.value == "failed" else "⏸️"
                print(f"   {status_icon} {stage_result.stage.value}: {stage_result.status.value}")
                if stage_result.duration:
                    print(f"      Duration: {stage_result.duration:.2f}s")
                if stage_result.error:
                    print(f"      Error: {stage_result.error}")
            print()
            
            # Step 5: Generate summary
            print("📄 Step 5: Document Summary")
            print("-" * 70)
            document = db.query(Document).filter(Document.id == document.id).first()
            if document:
                print(f"Document ID: {document.id}")
                print(f"Filename: {document.filename}")
                print(f"Status: {document.processing_status}")
                print(f"Total Pages: {document.total_pages}")
                print(f"Total Characters: {document.total_characters}")
                print(f"Total Words: {document.total_words}")
                print(f"Average OCR Confidence: {document.average_confidence:.2f}" if document.average_confidence else "")
            print()
            
            print("=" * 70)
            print("✅ Demo completed successfully!")
            print("=" * 70)
            print()
            print("Next steps:")
            print(f"  - View document: GET /api/documents/{document.id}")
            print(f"  - Get patterns: GET /api/patterns?document_id={document.id}")
            print(f"  - Generate report: GET /api/reports/{document.id}")
            print()
            
        else:
            print("❌ Processing failed!")
            print(f"Error: {result.error_message}")
            if result.stage_results:
                failed_stages = [s for s in result.stage_results if s.status.value == "failed"]
                for stage in failed_stages:
                    print(f"  Failed at: {stage.stage.value}")
                    if stage.error:
                        print(f"    Error: {stage.error}")
    
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        db.close()


def main():
    """Main entry point for demo script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CODEFINDER Demo Script")
    parser.add_argument(
        "--document",
        type=str,
        help="Path to document to process (default: uses sample document)"
    )
    
    args = parser.parse_args()
    
    # Run async demo
    asyncio.run(demo_document_processing(args.document))


if __name__ == "__main__":
    main()
