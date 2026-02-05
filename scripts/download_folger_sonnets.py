#!/usr/bin/env python3
"""
Download Folger Shakespeare Sonnets 1609
=========================================
CLI script to download high-resolution facsimile images from the 
Folger Shakespeare Library Digital Collections via IIIF.

Usage:
    python scripts/download_folger_sonnets.py
    python scripts/download_folger_sonnets.py --size 2000, --output data/sources/folger_sonnets_1609/
    python scripts/download_folger_sonnets.py --verify data/sources/folger_sonnets_1609/

Source: https://digitalcollections.folger.edu/bib169144-164315
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.folger_iiif_client import FolgerIIIFClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_bar(current: int, total: int, label: str, bar_length: int = 40):
    """Display a progress bar in the terminal."""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    sys.stdout.write(f'\r[{bar}] {current}/{total} - {label[:50]:<50}')
    sys.stdout.flush()
    if current == total:
        print()  # Newline at end


def download_command(args):
    """Download all pages from Folger Digital Collections."""
    output_dir = Path(args.output)
    
    print("=" * 70)
    print("Folger Shakespeare Library - Sonnets Quarto 1609 Downloader")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Image size: {args.size}")
    print(f"Image format: {args.format}")
    print()
    
    client = FolgerIIIFClient(cache_dir=output_dir)
    
    # Fetch manifest first
    print("📜 Fetching IIIF manifest...")
    manifest = client.fetch_manifest(use_cache=not args.force)
    
    pages = client.get_page_list()
    print(f"📖 Found {len(pages)} pages in the quarto")
    print()
    
    # Show first few pages
    print("Pages to download:")
    for page in pages[:5]:
        print(f"  • {page.sort_order:2d}. {page.label}")
    if len(pages) > 5:
        print(f"  ... and {len(pages) - 5} more")
    print()
    
    # Confirm download
    if not args.yes:
        confirm = input("Proceed with download? [Y/n] ").strip().lower()
        if confirm and confirm != 'y':
            print("Download cancelled.")
            return 1
    
    # Download with progress
    print("\n📥 Downloading pages...")
    result = client.download_all_pages(
        output_dir=output_dir,
        size=args.size,
        format=args.format,
        progress_callback=progress_bar
    )
    
    print()
    print("=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"  Total pages: {result['download']['total_pages']}")
    print(f"  Downloaded:  {result['download']['downloaded']}")
    if result['download']['errors']:
        print(f"  Errors:      {len(result['download']['errors'])}")
        for err in result['download']['errors']:
            print(f"    - {err['page']}: {err['error']}")
    print()
    print(f"Files saved to: {output_dir}")
    print(f"  • manifest.json       (IIIF manifest)")
    print(f"  • source_metadata.json (provenance)")
    print(f"  • checksums.sha256    (integrity verification)")
    
    return 0


def verify_command(args):
    """Verify downloaded files against checksums."""
    source_dir = Path(args.verify)
    
    if not source_dir.exists():
        print(f"Error: Directory not found: {source_dir}")
        return 1
    
    print("=" * 70)
    print("Verifying Folger Sonnets Download")
    print("=" * 70)
    print(f"Directory: {source_dir}")
    print()
    
    client = FolgerIIIFClient()
    
    try:
        results = client.verify_checksums(source_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    valid = sum(1 for v in results.values() if v)
    invalid = [f for f, v in results.items() if not v]
    
    print(f"✓ Valid files:   {valid}/{len(results)}")
    
    if invalid:
        print(f"✗ Invalid files: {len(invalid)}")
        for f in invalid:
            print(f"    - {f}")
        return 1
    else:
        print("\n✓ All checksums verified successfully!")
        return 0


def list_command(args):
    """List pages in the manifest without downloading."""
    client = FolgerIIIFClient()
    
    print("Fetching manifest...")
    client.fetch_manifest()
    pages = client.get_page_list()
    
    print()
    print(f"{'#':<4} {'Label':<50} {'Dimensions':<15}")
    print("-" * 70)
    
    for page in pages:
        dims = f"{page.width}x{page.height}"
        print(f"{page.sort_order:<4} {page.label:<50} {dims:<15}")
    
    print()
    print(f"Total: {len(pages)} pages")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Download Shakespeare Sonnets 1609 from Folger Digital Collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Download full resolution to default location
  %(prog)s --size 2000,                 # Download max 2000px width
  %(prog)s --list                       # List pages without downloading
  %(prog)s --verify data/sources/folger_sonnets_1609/  # Verify checksums
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        default="data/sources/folger_sonnets_1609",
        help="Output directory for downloaded images (default: data/sources/folger_sonnets_1609)"
    )
    
    parser.add_argument(
        "--size", "-s",
        default="full",
        help="IIIF size parameter: 'full' (native ~7000px), '2000,' (max 2000px), '1000,' (max 1000px)"
    )
    
    parser.add_argument(
        "--format", "-f",
        default="jpg",
        choices=["jpg", "png", "tif"],
        help="Output image format (default: jpg)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of manifest (ignore cache)"
    )
    
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    parser.add_argument(
        "--verify",
        metavar="DIR",
        help="Verify checksums of downloaded files in DIR"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List pages in manifest without downloading"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        return verify_command(args)
    elif args.list:
        return list_command(args)
    else:
        return download_command(args)


if __name__ == "__main__":
    sys.exit(main())
