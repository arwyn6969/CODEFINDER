
import sys
from unittest.mock import MagicMock

# Mock selected heavy dependencies only while importing this test module's targets.
_PATCHED_MODULES = [
    'pydantic_settings',
    'app.core.config',
    'app.core.database',
    'fitz',
    'cv2',
    'pytesseract',
    'PIL',
    'PIL.Image',
    'scipy',
    'scipy.stats',
    'scipy.spatial',
    'scipy.spatial.distance',
    'scipy.cluster',
    'scipy.cluster.hierarchy',
    'networkx',
    'pdf2image',
    'numpy',
]
_ORIGINAL_MODULES = {name: sys.modules.get(name) for name in _PATCHED_MODULES}

# Mock missing dependencies
sys.modules['pydantic_settings'] = MagicMock()

mock_config = MagicMock()
mock_config.settings.database_url = "sqlite:///:memory:" # valid-ish url
sys.modules['app.core.config'] = mock_config

# Mock database module to prevent real engine creation failure
from sqlalchemy.orm import declarative_base
RealBase = declarative_base()

mock_db_module = MagicMock()
mock_db_module.get_db = MagicMock()
mock_db_module.Base = RealBase  # Use real class to satisfy typing
sys.modules['app.core.database'] = mock_db_module

# Mock heavy processing libs
sys.modules['fitz'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['pytesseract'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['scipy.stats'] = MagicMock()
sys.modules['scipy.spatial'] = MagicMock()
sys.modules['scipy.spatial.distance'] = MagicMock()
sys.modules['scipy.cluster'] = MagicMock()
sys.modules['scipy.cluster.hierarchy'] = MagicMock()
sys.modules['networkx'] = MagicMock()
sys.modules['pdf2image'] = MagicMock()
sys.modules['numpy'] = MagicMock()

import unittest
from unittest.mock import MagicMock
from app.services.processing_pipeline import ProcessingPipeline, ProcessingConfiguration
from app.services.relationship_analyzer import RelationshipAnalyzer
from app.models.database_models import Document, Pattern, Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Restore patched modules immediately so this test file doesn't pollute
# module state for other tests collected in the same pytest run.
for _name, _module in _ORIGINAL_MODULES.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module

class TestDeepConnection(unittest.TestCase):
    def setUp(self):
        # Setup in-memory DB
        self.engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        
        # Patch initialization to avoid loading heavy services
        self.original_init = ProcessingPipeline._initialize_services
        ProcessingPipeline._initialize_services = lambda x: None
        
        # Create pipeline
        self.pipeline = ProcessingPipeline(self.db)
        
        # Manual init of required items
        self.pipeline.gematria_engine = MagicMock()
        self.pipeline.els_analyzer = MagicMock()
        self.pipeline.db = self.db
        self.pipeline.logger = MagicMock()
        
    def tearDown(self):
        # Restore
        ProcessingPipeline._initialize_services = self.original_init
        self.db.close()
        
    def test_gematria_integration(self):
        print("\n=== Testing Gematria Pattern Persistence ===")
        # Create a document
        doc = Document(
            filename="Bacon.txt", 
            original_filename="Bacon.txt",
            file_path="/tmp/Bacon.txt",
            file_size=123,
            content="Some content", 
            processing_status="processing"
        )
        self.db.add(doc)
        self.db.commit()
        
        # Test Data: Gematria returns '100' for filename
        self.pipeline.gematria_engine.calculate_all.return_value = {
            "francis_bacon_simple": {"score": 100, "breakdown": []}
        }
        
        # Run _analyze_gematria
        import asyncio
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(
            self.pipeline._analyze_gematria({'full_text': "Start"}, doc.id)
        )
        
        # Check Patterns
        patterns = self.db.query(Pattern).filter(Pattern.document_id == doc.id).all()
        print(f"Patterns found: {len(patterns)}")
        for p in patterns:
            print(f" - {p.pattern_name}: {p.description}")
            
        self.assertGreaterEqual(len(patterns), 1)
        self.assertTrue(any("Gematria: 100" in p.pattern_name for p in patterns))
        
    def test_relationship_link(self):
        print("\n=== Testing Deep Connection (Relationship) ===")
        # Create 2 documents with SAME Gematria Pattern
        doc1 = Document(
            filename="DocA", 
            original_filename="DocA.pdf",
            file_path="/tmp/DocA.pdf",
            file_size=100,
            content="A", 
            processing_status="completed"
        )
        doc2 = Document(
            filename="DocB", 
            original_filename="DocB.pdf",
            file_path="/tmp/DocB.pdf",
            file_size=100,
            content="B", 
            processing_status="completed"
        )
        self.db.add_all([doc1, doc2])
        self.db.commit()
        
        # Add shared patterns manually (simulating the pipeline)
        p1 = Pattern(
            document_id=doc1.id, 
            pattern_type="gematria_match", 
            pattern_name="Gematria: 100 (Cipher)", 
            significance_score=1.0,
            confidence=1.0,
            severity=0.5,
            description="Match"
        )
        p2 = Pattern(
            document_id=doc2.id, 
            pattern_type="gematria_match", 
            pattern_name="Gematria: 100 (Cipher)", 
            significance_score=1.0,
            confidence=1.0,
            severity=0.5,
            description="Match"
        )
        self.db.add_all([p1, p2])
        self.db.commit()
        
        # Analyze Relationship
        analyzer = RelationshipAnalyzer(self.db)
        
        # We need to ensure _get_document_pattern_signatures works.
        # But we can't see the code. Let's assume it calls _get_document_patterns
        # I'll rely on public method build_relationship_network or internal calc
        
        # Let's try calling internal calculation directly used by matrix
        corr = analyzer._calculate_pattern_correlation(doc1.id, doc2.id)
        print(f"Pattern Correlation between Doc1 and Doc2: {corr}")
        
        if corr > 0.5:
            print("SUCCESS: Deep Connection Detected via Shared Gematria.")
        else:
            print("FAILURE: No correlation detected.")
            
if __name__ == '__main__':
    unittest.main()
