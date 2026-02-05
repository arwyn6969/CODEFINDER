#!/usr/bin/env python3
"""
Download Aspley version of Shakespeare's Sonnets (STC 22353) from Folger Digital Collections.
This is a separate copy from the Wright version (STC 22353a) for comparative analysis.
"""

import sys
import json
import hashlib
import urllib.request
from pathlib import Path
from urllib.parse import quote

# IIIF Manifest for Aspley version
MANIFEST_URL = "https://digitalcollections.folger.edu/node/29467/manifest"
OUTPUT_DIR = Path("data/sources/folger_sonnets_1609_aspley")

def download_aspley_sonnets(output_dir: Path, size: str = "2000,"):
    """Download all pages from the Aspley version manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Downloading Aspley Version (STC 22353)")
    print("=" * 70)
    print(f"Manifest: {MANIFEST_URL}")
    print(f"Output: {output_dir}")
    print()
    
    # Fetch manifest
    print("📜 Fetching IIIF manifest...")
    with urllib.request.urlopen(MANIFEST_URL) as response:
        manifest = json.loads(response.read().decode('utf-8'))
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Saved manifest to {manifest_path}")
    
    # Extract canvases from sequences
    canvases = manifest.get("sequences", [{}])[0].get("canvases", [])
    print(f"📖 Found {len(canvases)} pages")
    print()
    
    # Sort canvases by sort order if available
    def get_sort_order(canvas):
        for meta in canvas.get("metadata", []):
            if meta.get("label") == "Sort order":
                try:
                    return int(meta["value"])
                except (ValueError, KeyError):
                    pass
        return 999
    
    canvases_sorted = sorted(canvases, key=get_sort_order)
    
    downloaded = 0
    errors = []
    checksums = {}
    
    print("📥 Downloading pages...")
    for idx, canvas in enumerate(canvases_sorted):
        # Get page label
        label = canvas.get("label", f"page_{idx + 1}")
        
        # Sanitize label for filename
        safe_label = label.replace(" ", "_").replace("–", "-").replace("/", "_")
        safe_label = "".join(c for c in safe_label if c.isalnum() or c in "_-.")
        filename = f"{idx + 1:03d}_{safe_label}.jpg"
        
        # Get image URL from canvas
        images = canvas.get("images", [])
        if not images:
            errors.append({"page": label, "error": "No images in canvas"})
            continue
        
        resource = images[0].get("resource", {})
        service = resource.get("service", {})
        
        if "@id" in service:
            # Use IIIF service to get specific size
            base_url = service["@id"]
            image_url = f"{base_url}/full/{size}/0/default.jpg"
        else:
            # Use direct URL
            image_url = resource.get("@id", "")
        
        if not image_url:
            errors.append({"page": label, "error": "No image URL found"})
            continue
        
        # Download image
        output_path = output_dir / filename
        try:
            print(f"  [{idx + 1}/{len(canvases_sorted)}] {label[:50]}...", end=" ", flush=True)
            
            req = urllib.request.Request(image_url, headers={"User-Agent": "CODEFINDER/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = response.read()
            
            with open(output_path, "wb") as f:
                f.write(data)
            
            # Calculate checksum
            checksums[filename] = hashlib.sha256(data).hexdigest()
            
            print(f"✓ ({len(data) // 1024} KB)")
            downloaded += 1
            
        except Exception as e:
            print(f"✗ {e}")
            errors.append({"page": label, "error": str(e)})
    
    # Save checksums
    checksum_path = output_dir / "checksums.sha256"
    with open(checksum_path, "w") as f:
        for fname, sha in sorted(checksums.items()):
            f.write(f"{sha}  {fname}\n")
    
    # Save metadata
    metadata = {
        "source": {
            "name": "Folger Shakespeare Library",
            "catalog_id": "STC 22353",
            "variant": "Aspley",
            "url": "https://digitalcollections.folger.edu/node/29467",
            "manifest": MANIFEST_URL,
            "rights": "https://rightsstatements.org/page/NoC-US/1.0/"
        },
        "download": {
            "total_pages": len(canvases_sorted),
            "downloaded": downloaded,
            "skipped": len(errors),
            "errors": errors,
            "size_param": size,
            "format": "jpg"
        }
    }
    
    metadata_path = output_dir / "source_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"  Total pages:  {len(canvases_sorted)}")
    print(f"  Downloaded:   {downloaded}")
    if errors:
        print(f"  Errors:       {len(errors)}")
        for err in errors:
            print(f"    - {err['page']}: {err['error']}")
    print()
    print(f"Files saved to: {output_dir}")
    
    return metadata


if __name__ == "__main__":
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DIR
    download_aspley_sonnets(output_dir)
