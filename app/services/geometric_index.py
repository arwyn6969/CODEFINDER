"""
Geometric Indexing Service
Provides optimized spatial queries using KD-Trees to accelerate geometric pattern detection.
"""
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    index: int  # Original index reference

class GeometricIndex:
    """
    Spatial index for fast geometric queries.
    Wraps scipy.spatial.cKDTree to provide domain-specific lookups.
    """
    
    def __init__(self, points: List[Tuple[float, float]]):
        """
        Initialize the index with a list of (x, y) coordinates.
        """
        self.points = np.array(points)
        self.tree = cKDTree(self.points)
        self.n_points = len(points)
        
    def find_nearest_neighbors(self, query_point: Tuple[float, float], k: int = 3) -> List[Tuple[float, int]]:
        """
        Find k-nearest neighbors for a point.
        Returns list of (distance, original_index).
        """
        distances, indices = self.tree.query(query_point, k=k)
        # Handle single point vs multiple points return format
        if k == 1:
            return [(distances, indices)]
        return list(zip(distances, indices))
        
    def find_pairs_within_distance(self, max_distance: float) -> List[Tuple[int, int]]:
        """
        Find all pairs of points within a specified distance.
        Returns list of (idx1, idx2).
        """
        return list(self.tree.query_pairs(r=max_distance))
        
    def find_points_in_annulus(self, center_idx: int, min_radius: float, max_radius: float) -> List[int]:
        """
        Find all points in a ring (annulus) around a center point.
        Useful for finding points at a specific distance (within tolerance).
        """
        center_point = self.points[center_idx]
        
        # Query for outer radius
        indices_outer = self.tree.query_ball_point(center_point, r=max_radius)
        
        # Filter manually or query inner if efficiency demands (usually filtering is fast for small N)
        # For strict annulus: query inner and usage set difference? 
        # KDTree doesn't support annulus natively, but set difference works:
        if min_radius > 0:
            indices_inner = self.tree.query_ball_point(center_point, r=min_radius)
            return list(set(indices_outer) - set(indices_inner))
        
        return indices_outer

    def find_potential_triangles(self, center_idx: int, radius: float, tolerance: float) -> List[Tuple[int, int, int]]:
        """
        Find potential triangles where two points are equidistant from center (Isosceles candidates).
        """
        # Get points at distance 'radius' (+/- tolerance)
        candidates = self.find_points_in_annulus(center_idx, radius - tolerance, radius + tolerance)
        
        triangles = []
        # Check pairs of candidates
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                # This forms a triangle (center, p1, p2)
                # We can filter by angles here if needed or return for further processing
                triangles.append((center_idx, candidates[i], candidates[j]))
                
        return triangles
