import numpy as np
from scipy.spatial import cKDTree
from app.services.geometric_index import GeometricIndex

def test_geometric_index():
    print("Testing GeometricIndex...")
    
    # Create random points
    points = [(0, 0), (1, 1), (10, 10), (0.1, 0.1)]
    index = GeometricIndex(points)
    
    # Test Nearest Neighbors
    print("Finding neighbors for (0,0)...")
    neighbors = index.find_nearest_neighbors((0,0), k=3)
    print(f"Neighbors: {neighbors}")
    # Expected: (0,0) [dist 0], (0.1, 0.1) [dist 0.14], (1,1) [dist 1.41]
    
    distances = [n[0] for n in neighbors]
    indices = [n[1] for n in neighbors]
    
    assert 0 in indices  # Should find itself
    assert 3 in indices  # Should find (0.1, 0.1) - index 3
    
    print("Finding pairs within distance 2.0...")
    pairs = index.find_pairs_within_distance(2.0)
    print(f"Pairs: {pairs}")
    # Expected: {(0, 3), (1, 3), (0, 1)} approx
    
    print("SUCCESS: GeometricIndex is functioning.")

if __name__ == "__main__":
    try:
        test_geometric_index()
    except Exception as e:
        print(f"FAILURE: {e}")
