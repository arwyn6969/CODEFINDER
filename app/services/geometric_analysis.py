
import numpy as np
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class GeometricPoint:
    x: float
    y: float
    character: str
    confidence: float

class GeometricAligner:
    """
    Performs geometric alignment analysis between two sets of typographical points.
    Uses KD-Tree to find nearest neighbors and calculate RMSE (Root Mean Square Error).
    """
    
    def __init__(self, reference_points: List[GeometricPoint], target_points: List[GeometricPoint]):
        self.ref_points = reference_points
        self.target_points = target_points
        self.ref_coords = np.array([[p.x, p.y] for p in reference_points])
        self.target_coords = np.array([[p.x, p.y] for p in target_points])
        
    def _get_centroid(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.array([0.0, 0.0])
        return np.median(points, axis=0) # Use median to be robust to outliers

    def align_and_compare(self, max_distance: float = 10.0, auto_align: bool = True, auto_scale: bool = True, strict_match: bool = False) -> Dict[str, Any]:
        """
        Align target points to reference points and calculate error metrics.
        
        Args:
            max_distance: Maximum distance (pixels) to consider a valid match
            auto_align: If True, shifts target points to match geometric centroid of reference
            auto_scale: If True, scales target points to match width/height of reference
            strict_match: If True, requires characters to be identical.
        """
        ref_coords = self.ref_coords
        target_coords = self.target_coords
        
        translation = np.array([0.0, 0.0])
        scale_factor = 1.0
        
        if len(ref_coords) > 0 and len(target_coords) > 0:
            if auto_scale:
                # Calculate bounding box widths
                ref_min = np.min(ref_coords, axis=0)
                ref_max = np.max(ref_coords, axis=0)
                target_min = np.min(target_coords, axis=0)
                target_max = np.max(target_coords, axis=0)
                
                ref_width = ref_max[0] - ref_min[0]
                target_width = target_max[0] - target_min[0]
                
                if target_width > 0:
                    scale_factor = ref_width / target_width
                    # Apply scaling around the centroid (or top-left? centroid is safer)
                    target_centroid = self._get_centroid(target_coords)
                    target_coords = (target_coords - target_centroid) * scale_factor + target_centroid

            if auto_align:
                ref_centroid = self._get_centroid(ref_coords)
                target_centroid = self._get_centroid(target_coords) # Re-calc after scaling
                translation = ref_centroid - target_centroid
                target_coords = target_coords + translation
            
        # Build KD-Tree for reference points
        tree = KDTree(ref_coords)
        
        # Query nearest neighbors for all target points
        distances, indices = tree.query(target_coords, k=1, distance_upper_bound=max_distance)
        
        # Filter valid matches (distance < max_distance and not inf)
        valid_matches = []
        match_errors = []
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist == float('inf'):
                continue
                
            # Content verification logic could go here (e.g. check if char matches)
            # For pure geometric proof, we strictly look at layout
            
            target_pt = self.target_points[i]
            ref_pt = self.ref_points[idx]
            
            # Optional: Enforce character identity for stricter matching
            if not strict_match or target_pt.character == ref_pt.character:
                valid_matches.append((target_pt, ref_pt))
                match_errors.append(dist)
        
        match_count = len(match_errors)
        if match_count == 0:
            return {
                "rmse": float('inf'),
                "match_count": 0,
                "match_percentage": 0.0,
                "translation": translation.tolist()
            }
            
        # Calculate RMSE
        mse = np.mean(np.array(match_errors) ** 2)
        rmse = np.sqrt(mse)
        
        return {
            "rmse": rmse,
            "match_count": match_count,
            "total_target_points": len(self.target_points),
            "match_percentage": (match_count / len(self.target_points)) * 100,
            "mean_error": np.mean(match_errors),
            "std_error": np.std(match_errors),
            "min_error": np.min(match_errors),
            "max_error": np.max(match_errors),
            "translation": translation.tolist(),
            "scale_factor": float(scale_factor)
        }

    def optimize_alignment(self) -> Dict[str, Any]:
        """
        Uses scipy.optimize to find the best translation and scale to minimize RMSE.
        """
        from scipy.optimize import minimize
        
        # Initial guess: Centroid alignment + Width scaling
        ref_centroid = self._get_centroid(self.ref_coords)
        target_centroid = self._get_centroid(self.target_coords)
        
        # Estimate scale from widths
        ref_width = np.ptp(self.ref_coords[:, 0])
        target_width = np.ptp(self.target_coords[:, 0])
        initial_scale = ref_width / target_width if target_width > 0 else 1.0
        
        initial_dx = ref_centroid[0] - (target_centroid[0] * initial_scale)
        initial_dy = ref_centroid[1] - (target_centroid[1] * initial_scale)
        
        tree = KDTree(self.ref_coords)
        
        def objective(params):
            dx, dy, s = params
            # Transform target
            transformed = self.target_coords * s + np.array([dx, dy])
            # Query nearest neighbors
            dists, _ = tree.query(transformed, k=1)
            # Use sum of squared errors (robust mean)
            # Clip large errors to avoid outliers driving the optimization
            dists = np.clip(dists, 0, 100) 
            return np.mean(dists**2)
            
        result = minimize(objective, x0=[initial_dx, initial_dy, initial_scale], method='Nelder-Mead', tol=0.1)
        
        dx, dy, s = result.x
        rmse = np.sqrt(result.fun)
        
        return {
            "optimized_dx": dx,
            "optimized_dy": dy,
            "optimized_scale": s,
            "rmse": rmse,
            "success": result.success
        }
