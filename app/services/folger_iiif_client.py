"""
Folger Digital Collections IIIF Client
=======================================
Client for accessing Shakespeare Sonnets 1609 from the Folger Shakespeare Library
via the IIIF (International Image Interoperability Framework) API.

Source: https://digitalcollections.folger.edu/bib169144-164315
IIIF Manifest: https://digitalcollections.folger.edu/node/70076/manifest

The 1609 Quarto of Shakespeare's Sonnets (STC 22353a) is available in high-resolution
TIFF format with scholarly provenance and public domain licensing.
"""

import hashlib
import json
import logging
import re
import requests
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from urllib.parse import quote, unquote

logger = logging.getLogger(__name__)


@dataclass
class PageMetadata:
    """Metadata for a single page from the IIIF manifest."""
    canvas_id: str
    label: str
    sort_order: int
    width: int
    height: int
    image_service_url: str
    thumbnail_url: str
    digital_file_name: str
    
    @property
    def safe_filename(self) -> str:
        """Generate a safe filename from the label."""
        # Convert label to safe filename
        safe = re.sub(r'[^\w\s-]', '', self.label.lower())
        safe = re.sub(r'[-\s]+', '_', safe).strip('_')
        return f"{self.sort_order:03d}_{safe}"


@dataclass  
class FolgerSource:
    """Metadata about the Folger source collection."""
    manifest_url: str = "https://digitalcollections.folger.edu/node/70076/manifest"
    catalog_url: str = "https://digitalcollections.folger.edu/bib169144-164315"
    catalog_record: str = "https://catalog.folger.edu/record/169144"
    call_number: str = "STC 22353a"
    title: str = "Shake-speares sonnets : neuer before imprinted"
    date: str = "1609"
    printer: str = "Eld, George, -1624"
    author: str = "Shakespeare, William, 1564-1616"
    rights: str = "https://rightsstatements.org/page/NoC-US/1.0/"
    total_pages: int = 52


class IIIFClient:
    """
    Generic Client for IIIF Image API.
    
    Provides access to high-resolution facsimile images via IIIF Image API v2/v3.
    Defaults to Folger Shakespeare Library Sonnets 1609 if no manifest provided.
    
    Example usage:
        client = IIIFClient(manifest_url="...")
        pages = client.get_page_list()
        client.download_all_pages(Path("output_dir/"))
    """
    
    # IIIF Image API size parameters
    SIZE_FULL = "full"          # Native resolution
    SIZE_2000 = "2000,"         # Max 2000px width
    SIZE_1000 = "1000,"         # Max 1000px width  
    SIZE_THUMBNAIL = "480,"     # Thumbnail size
    
    def __init__(self, manifest_url: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize the IIIF client.
        
        Args:
            manifest_url: URL to IIIF manifest. If None, defaults to Folger Sonnets.
            cache_dir: Optional directory for caching manifest and metadata
        """
        self.source = FolgerSource() # Keep for metadata defaults/legacy
        self.manifest_url = manifest_url if manifest_url else self.source.manifest_url
        self.cache_dir = cache_dir
        self._manifest: Optional[Dict[str, Any]] = None
        self._pages: Optional[List[PageMetadata]] = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CODEFINDER-Research/1.0 (Forensic Print Analysis)'
        })
    
    def fetch_manifest(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Fetch the IIIF Presentation API manifest.
        
        Args:
            use_cache: If True and cache exists, use cached manifest
            
        Returns:
            Parsed IIIF manifest as dictionary
        """
        # Check cache first
        if use_cache and self.cache_dir:
            cache_hash = hashlib.md5(self.manifest_url.encode()).hexdigest()
            cache_path = self.cache_dir / f"manifest_{cache_hash}.json"
            if cache_path.exists():
                logger.info(f"Loading cached manifest from {cache_path}")
                with open(cache_path) as f:
                    self._manifest = json.load(f)
                return self._manifest
        
        # Fetch from Source
        logger.info(f"Fetching IIIF manifest from {self.manifest_url}")
        try:
            response = self.session.get(self.manifest_url, timeout=30)
            response.raise_for_status()
            self._manifest = response.json()
        except Exception as e:
            logger.error(f"Failed to fetch manifest: {e}")
            raise
        
        # Cache if directory specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_hash = hashlib.md5(self.manifest_url.encode()).hexdigest()
            cache_path = self.cache_dir / f"manifest_{cache_hash}.json"
            with open(cache_path, 'w') as f:
                json.dump(self._manifest, f, indent=2)
            logger.info(f"Cached manifest to {cache_path}")
        
        return self._manifest
    
    def get_page_list(self) -> List[PageMetadata]:
        """
        Parse manifest and return list of page metadata.
        
        Returns:
            List of PageMetadata objects for all pages in the quarto
        """
        if self._pages is not None:
            return self._pages
            
        if self._manifest is None:
            self.fetch_manifest()
        
        pages = []
        sequences = self._manifest.get("sequences", [])
        if not sequences:
            raise ValueError("No sequences found in manifest")
        
        canvases = sequences[0].get("canvases", [])
        logger.info(f"Found {len(canvases)} canvases in manifest")
        
        for i, canvas in enumerate(canvases):
            # Extract metadata from canvas
            canvas_id = canvas.get("@id", "")
            label = canvas.get("label", f"page_{i+1}")
            width = canvas.get("width", 0)
            height = canvas.get("height", 0)
            
            # Get image service URL from first image
            images = canvas.get("images", [])
            if not images:
                # Some manifests put service directly on canvas (less common in v2)
                # or structure it differently. Fallback logic could go here.
                # For now, skip empty canvases unless we find a service block elsehwere.
                continue
                
            resource = images[0].get("resource", {})
            service = resource.get("service", {})
            
            # Handle list of services or single service dict
            if isinstance(service, list):
                if service:
                    image_service_url = service[0].get("@id", "")
                else:
                    image_service_url = ""
            else:
                image_service_url = service.get("@id", "")

            # Get thumbnail
            thumbnail = canvas.get("thumbnail", {})
            if isinstance(thumbnail, str): # sometimes just a URL string
                thumbnail_url = thumbnail
            else:
                thumbnail_url = thumbnail.get("@id", "")
            
            # Extract metadata fields
            metadata_list = canvas.get("metadata", [])
            metadata = {}
            if isinstance(metadata_list, list):
                for m in metadata_list:
                    if isinstance(m, dict):
                         # Handle label/value pairs which can be strings or lists
                         l = m.get("label", "")
                         v = m.get("value", "")
                         if isinstance(l, list): l = " ".join(str(x) for x in l)
                         if isinstance(v, list): v = " ".join(str(x) for x in v)
                         metadata[str(l)] = str(v)

            # Try to get specific metadata with fallbacks
            # Folger has "Sort order", others might not
            try:
                if "Sort order" in metadata:
                    sort_order = int(metadata["Sort order"])
                else:
                    sort_order = i + 1
            except (ValueError, TypeError):
                sort_order = i + 1

            digital_file_name = metadata.get("Digital image file name", "")
            
            page = PageMetadata(
                canvas_id=canvas_id,
                label=str(label),
                sort_order=sort_order,
                width=width,
                height=height,
                image_service_url=image_service_url,
                thumbnail_url=thumbnail_url,
                digital_file_name=digital_file_name
            )
            pages.append(page)
        
        # Sort by sort_order
        pages.sort(key=lambda p: p.sort_order)
        self._pages = pages
        
        logger.info(f"Parsed {len(pages)} pages from manifest")
        return pages
    
    def get_image_url(self, page: PageMetadata, 
                      size: str = "full",
                      format: str = "jpg",
                      quality: str = "default") -> str:
        """
        Construct IIIF Image API URL for a page.
        
        Args:
            page: PageMetadata object
            size: IIIF size parameter (e.g., "full", "2000,", "1000,")
            format: Output format ("jpg", "png", "tif")
            quality: Quality parameter ("default", "color", "gray")
            
        Returns:
            Full URL for image download
        """
        # IIIF Image API URL pattern: {scheme}://{server}{/prefix}/{identifier}/{region}/{size}/{rotation}/{quality}.{format}
        base_url = page.image_service_url
        return f"{base_url}/full/{size}/0/{quality}.{format}"
    
    def download_page(self, page: PageMetadata,
                      output_dir: Path,
                      size: str = "full",
                      format: str = "jpg",
                      verify_checksum: bool = True) -> Path:
        """
        Download a single page image.
        
        Args:
            page: PageMetadata object
            output_dir: Directory to save the image
            size: IIIF size parameter
            format: Output format
            verify_checksum: If True, compute and save SHA-256 checksum
            
        Returns:
            Path to the downloaded file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{page.safe_filename}.{format}"
        output_path = output_dir / filename
        
        # Skip if already downloaded
        if output_path.exists():
            logger.debug(f"Page already exists: {output_path}")
            return output_path
        
        url = self.get_image_url(page, size=size, format=format)
        logger.info(f"Downloading {page.label} -> {filename}")
        
        response = self.session.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        # Write to file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Compute checksum
        if verify_checksum:
            checksum = self._compute_sha256(output_path)
            logger.debug(f"SHA-256: {checksum}")
        
        return output_path
    
    def download_all_pages(self, output_dir: Path,
                           size: str = "full",
                           format: str = "jpg",
                           progress_callback=None) -> Dict[str, Any]:
        """
        Download all pages from the Sonnets quarto.
        
        Args:
            output_dir: Directory to save images
            size: IIIF size parameter
            format: Output format
            progress_callback: Optional callback(current, total, page_label)
            
        Returns:
            Summary dict with download statistics and checksums
        """
        pages = self.get_page_list()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checksums = {}
        downloaded = 0
        skipped = 0
        errors = []
        
        for i, page in enumerate(pages):
            try:
                if progress_callback:
                    progress_callback(i + 1, len(pages), page.label)
                
                output_path = self.download_page(
                    page, output_dir, size=size, format=format
                )
                
                if output_path.exists():
                    checksums[output_path.name] = self._compute_sha256(output_path)
                    downloaded += 1
                    
            except Exception as e:
                logger.error(f"Failed to download {page.label}: {e}")
                errors.append({"page": page.label, "error": str(e)})
        
        # Save checksums
        checksum_path = output_dir / "checksums.sha256"
        with open(checksum_path, 'w') as f:
            for filename, checksum in sorted(checksums.items()):
                f.write(f"{checksum}  {filename}\n")
        
        # Save manifest cache
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists() and self._manifest:
            with open(manifest_path, 'w') as f:
                json.dump(self._manifest, f, indent=2)
        
        # Create source metadata
        metadata = {
            "source": {
                "name": "Folger Shakespeare Library",
                "catalog_id": self.source.call_number,
                "url": self.source.catalog_url,
                "manifest": self.source.manifest_url,
                "rights": self.source.rights,
            },
            "download": {
                "total_pages": len(pages),
                "downloaded": downloaded,
                "skipped": skipped,
                "errors": errors,
                "size_param": size,
                "format": format,
            }
        }
        
        metadata_path = output_dir / "source_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Download complete: {downloaded}/{len(pages)} pages")
        return metadata
    
    def verify_checksums(self, source_dir: Path) -> Dict[str, bool]:
        """
        Verify downloaded files against stored checksums.
        
        Args:
            source_dir: Directory containing downloaded images
            
        Returns:
            Dict mapping filename to verification status
        """
        checksum_path = source_dir / "checksums.sha256"
        if not checksum_path.exists():
            raise FileNotFoundError(f"No checksum file found at {checksum_path}")
        
        results = {}
        with open(checksum_path) as f:
            for line in f:
                expected_hash, filename = line.strip().split("  ", 1)
                file_path = source_dir / filename
                
                if not file_path.exists():
                    results[filename] = False
                    continue
                
                actual_hash = self._compute_sha256(file_path)
                results[filename] = (actual_hash == expected_hash)
        
        valid = sum(1 for v in results.values() if v)
        logger.info(f"Checksum verification: {valid}/{len(results)} files valid")
        return results
    
    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def download_at_max_resolution(self, output_dir: Path,
                                    format: str = "jpg",
                                    force: bool = False) -> Dict[str, Any]:
        """
        Download (or re-download) all pages at native full resolution.
        
        This is critical for sort-level analysis — the default 2000px Aspley
        downloads are too low for wear-mark comparison.
        
        Args:
            output_dir: Directory for full-resolution images
            format: Output format ('jpg' or 'tif'; tif is lossless but large)
            force: If True, re-download even if files already exist
            
        Returns:
            Download summary dict
        """
        pages = self.get_page_list()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        for page in pages:
            filename = f"{page.safe_filename}.{format}"
            output_path = output_dir / filename
            
            if output_path.exists() and not force:
                continue
            
            url = self.get_image_url(page, size=self.SIZE_FULL, format=format)
            logger.info(f"Downloading MAX-RES: {page.label} -> {filename}")
            
            try:
                response = self.session.get(url, timeout=300, stream=True)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded += 1
                
            except Exception as e:
                logger.error(f"Failed to download {page.label} at max res: {e}")
        
        logger.info(f"Max-resolution download: {downloaded} new files to {output_dir}")
        return {"downloaded": downloaded, "total_pages": len(pages), "format": format}

    @classmethod
    def download_source_by_config(cls, source_key: str,
                                   config_path: str = None,
                                   max_resolution: bool = True) -> Dict[str, Any]:
        """
        Download a source defined in config.yaml.
        
        Args:
            source_key: Key from config.yaml (e.g., 'folger_iiif', 'folger_iiif_aspley')
            config_path: Path to config.yaml (default: data/sources/config.yaml)
            max_resolution: If True, download at full native resolution
            
        Returns:
            Download summary dict
        """
        import yaml
        
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "data" / "sources" / "config.yaml"
        else:
            config_path = Path(config_path)
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        sources = config.get("sources", {})
        if source_key not in sources:
            available = list(sources.keys())
            raise ValueError(f"Unknown source '{source_key}'. Available: {available}")
        
        source_config = sources[source_key]
        
        if source_config.get("type") != "iiif_images":
            raise ValueError(f"Source '{source_key}' is not an IIIF source (type: {source_config.get('type')})")
        
        manifest_url = source_config.get("provenance", {}).get("manifest")
        if not manifest_url:
            raise ValueError(f"No manifest URL in config for '{source_key}'")
        
        # Create client with custom manifest URL
        client = cls(manifest_url=manifest_url)
        
        # Determine output directory
        base_dir = config_path.parent
        output_dir = base_dir / source_config["path"]
        
        size = cls.SIZE_FULL if max_resolution else "2000,"
        
        return client.download_all_pages(output_dir, size=size)


# Convenience function for quick access
def get_folger_sonnets_client(cache_dir: Optional[Path] = None) -> IIIFClient:
    """Get a configured FolgerIIIFClient instance."""
    return IIIFClient(cache_dir=cache_dir)

# Backwards compatibility alias
FolgerIIIFClient = IIIFClient

