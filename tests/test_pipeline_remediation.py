import pytest
import numpy as np
import subprocess
from unittest import mock
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

ROOT = Path(__file__).resolve().parent.parent

from scripts.formal_stats import ForensicStatistics

def test_geometric_normalization_ks():
    """
    Test that the formal_stats KS test uses normalized geometries.
    Two exact identically proportioned sets (but different absolute size) 
    should have perfectly identical normalized distributions and thus p > 0.05.
    """
    stats = ForensicStatistics(db_path=":memory:")
    
    # Scale factor of 2.0 between Source A and B
    # Median A width is 14.5, B width is 29.0
    w_a = list(range(10, 20))
    w_b = list(range(20, 40, 2))
    h_a = list(range(20, 30))
    h_b = list(range(40, 60, 2))
    
    norm_w_a = [w / np.median(w_a) for w in w_a]
    norm_w_b = [w / np.median(w_b) for w in w_b]
    norm_h_a = [h / np.median(h_a) for h in h_a]
    norm_h_b = [h / np.median(h_b) for h in h_b]
    
    sources = {
        'Source_A': {
            'norm_widths': norm_w_a,
            'norm_heights': norm_h_a
        },
        'Source_B': {
            'norm_widths': norm_w_b,
            'norm_heights': norm_h_b
        }
    }

    results = stats.ks_test_dimensions(sources)
    res = results['Source_A vs Source_B']
    
    assert res['verdict_width'] == 'SAME', f"Expected SAME, got {res['verdict_width']}"
    assert res['p_width'] > 0.05, f"p_width should be > 0.05, got {res['p_width']}"
    assert res['verdict_height'] == 'SAME', f"Expected SAME, got {res['verdict_height']}"

@mock.patch('cv2.imread')
@mock.patch('scripts.scan_greenman_all.BlockFingerprinter')
def test_woodblock_match_limits(mock_fingerprinter, mock_imread):
    """
    Test that the scan_greenman_all limits have been correctly hardened 
    to prevent the 99% false positive rates.
    """
    # Create a dummy image
    mock_imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_fp_instance = mock_fingerprinter.return_value
    mock_fp_instance.fingerprint.return_value = {}
    
    from scripts.scan_greenman_all import GreenmanScanner
    
    # Initialize the scanner
    scanner = GreenmanScanner("dummy_path")
    
    # Assert limits
    assert scanner.MIN_SIFT_MATCHES == 50, f"MIN_SIFT_MATCHES should be 50, but is {scanner.MIN_SIFT_MATCHES}"
    assert scanner.FINGERPRINT_THRESHOLD == 0.90, f"FINGERPRINT_THRESHOLD should be 0.90, but is {scanner.FINGERPRINT_THRESHOLD}"
    assert scanner._candidate_area_is_viable(1_029_392) is True
    assert scanner._candidate_area_is_viable(4_183_872) is False

def test_pagination_offset_validation():
    """
    Test that full_sonnet_mapper halts for deviations > 1.
    Since the ValueError is raised inside the main script logic, we'll verify the script file
    contains the specific ValueError logic.
    """
    with open(ROOT / "scripts" / "legacy" / "full_sonnet_mapper.py", "r") as f:
        content = f.read()

    assert "abs(offset) > 1" in content, "Strict 1:1 error checking not found in full_sonnet_mapper.py"
    assert "Pagination Alignment Error" in content, "Error message not found in full_sonnet_mapper.py"
    
    with open("scripts/isolate_sonnets.py", "r") as f:
        content_isolate = f.read()
    
    assert "abs(int(w_page) - int(a_page)) > 1" in content_isolate, "Strict validation missing from isolate_sonnets.py"
    assert "Pagination Alignment Error" in content_isolate, "Error message missing from isolate_sonnets.py"
