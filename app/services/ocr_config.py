"""
OCR Configuration
=================
Configuration management for OCR engines.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TesseractConfig:
    """Tesseract-specific configuration."""
    config: str = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    language: str = "eng"


@dataclass
class GeminiConfig:
    """Gemini-specific configuration."""
    model: str = "gemini-2.5-flash"
    preserve_long_s: bool = True
    detect_ligatures: bool = True
    rate_limit: int = 15  # requests per minute
    api_key: Optional[str] = None  # Uses env var if not set


@dataclass
class OCRConfig:
    """
    Complete OCR configuration.
    
    Can be loaded from YAML or created programmatically.
    """
    primary_engine: str = "auto"  # "gemini", "tesseract", or "auto"
    fallback_engine: str = "tesseract"
    
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    
    # Historical document settings
    preserve_long_s: bool = True
    detect_ligatures: bool = True
    save_anomalies: bool = True
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRConfig":
        """Create config from dictionary."""
        ocr_data = data.get("ocr", data)
        
        tesseract_data = ocr_data.get("tesseract", {})
        gemini_data = ocr_data.get("gemini", {})
        
        return cls(
            primary_engine=ocr_data.get("primary_engine", "auto"),
            fallback_engine=ocr_data.get("fallback_engine", "tesseract"),
            tesseract=TesseractConfig(
                config=tesseract_data.get("config", "--oem 3 --psm 6"),
                language=tesseract_data.get("language", "eng"),
            ),
            gemini=GeminiConfig(
                model=gemini_data.get("model", "gemini-2.5-flash"),
                preserve_long_s=gemini_data.get("preserve_long_s", True),
                detect_ligatures=gemini_data.get("detect_ligatures", True),
                rate_limit=gemini_data.get("rate_limit", 15),
                api_key=gemini_data.get("api_key"),
            ),
            preserve_long_s=ocr_data.get("preserve_long_s", True),
            detect_ligatures=ocr_data.get("detect_ligatures", True),
            save_anomalies=ocr_data.get("save_anomalies", True),
        )
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "OCRConfig":
        """Load config from YAML file."""
        if not YAML_AVAILABLE:
            logger.warning("PyYAML not installed, using defaults")
            return cls()
        
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.warning(f"Config file not found: {yaml_path}, using defaults")
            return cls()
        
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ocr": {
                "primary_engine": self.primary_engine,
                "fallback_engine": self.fallback_engine,
                "tesseract": {
                    "config": self.tesseract.config,
                    "language": self.tesseract.language,
                },
                "gemini": {
                    "model": self.gemini.model,
                    "preserve_long_s": self.gemini.preserve_long_s,
                    "detect_ligatures": self.gemini.detect_ligatures,
                    "rate_limit": self.gemini.rate_limit,
                },
                "preserve_long_s": self.preserve_long_s,
                "detect_ligatures": self.detect_ligatures,
                "save_anomalies": self.save_anomalies,
            }
        }


def load_ocr_config(config_path: Path = None) -> OCRConfig:
    """
    Load OCR configuration.
    
    Args:
        config_path: Path to YAML config file (optional)
        
    Returns:
        OCRConfig instance
    """
    if config_path:
        return OCRConfig.from_yaml(config_path)
    
    # Try default locations
    default_paths = [
        Path("config.yaml"),
        Path("data/sources/config.yaml"),
        Path.home() / ".codefinder" / "config.yaml",
    ]
    
    for path in default_paths:
        if path.exists():
            return OCRConfig.from_yaml(path)
    
    return OCRConfig()
