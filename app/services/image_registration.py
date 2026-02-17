"""
Image Registration Service for CODEFINDER
==========================================
Aligns page images from different copies of the same printed work
for pixel-level comparison and sort-level forensic analysis.

Uses OpenCV feature matching (ORB/SIFT) with RANSAC homography
to geometrically align pages despite differences in:
 - Resolution (Wright ~7000px vs Aspley ~2000px)
 - Scanning angle and distortion
 - Cropping and margins
 - Paper condition and discoloration

Usage:
    from app.services.image_registration import ImageRegistrationService
    
    service = ImageRegistrationService()
    result = service.register_pair(
        reference_path="data/sources/folger_sonnets_1609/010_b1_verso_b2_recto.jpg",
        target_path="data/sources/folger_sonnets_1609_aspley/010_leaf_B1_verso_-_leaf_B2_recto.jpg",
    )
    result.aligned_target.save("aligned_aspley_010.png")
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Result of aligning two page images."""
    reference_path: str
    target_path: str
    aligned_target_path: Optional[str] = None
    
    # Alignment quality metrics
    num_keypoints_ref: int = 0
    num_keypoints_target: int = 0
    num_matches: int = 0
    num_inliers: int = 0
    inlier_ratio: float = 0.0
    reprojection_error: float = 0.0
    
    # Resolution info
    reference_size: Tuple[int, int] = (0, 0)  # (width, height)
    target_original_size: Tuple[int, int] = (0, 0)
    target_aligned_size: Tuple[int, int] = (0, 0)
    scale_factor: float = 1.0
    
    # Homography matrix (3x3)
    homography: Optional[np.ndarray] = None
    
    # PIL images (populated in-memory, not persisted)
    aligned_target: Optional[Any] = field(default=None, repr=False)
    
    @property
    def is_good_alignment(self) -> bool:
        """Heuristic: alignment is good if inlier ratio > 0.25 and >10 inliers."""
        return self.num_inliers >= 10 and self.inlier_ratio > 0.25
    
    @property
    def quality_label(self) -> str:
        if self.num_inliers < 10:
            return "FAILED"
        if self.inlier_ratio < 0.25:
            return "POOR"
        if self.inlier_ratio < 0.50:
            return "FAIR"
        if self.inlier_ratio < 0.75:
            return "GOOD"
        return "EXCELLENT"


class ImageRegistrationService:
    """
    Aligns page images from different copies using feature-based registration.
    
    The reference image is the higher-resolution copy (typically Wright).
    The target image is warped to match the reference geometry.
    """
    
    # Feature detector options
    DETECTOR_ORB = "orb"
    DETECTOR_SIFT = "sift"
    
    def __init__(self, 
                 detector: str = "orb",
                 max_features: int = 5000,
                 match_ratio: float = 0.75,
                 ransac_threshold: float = 5.0,
                 output_dir: Optional[str] = None):
        """
        Args:
            detector: Feature detector to use ('orb' or 'sift')
            max_features: Maximum features to detect per image
            match_ratio: Lowe's ratio test threshold (lower = stricter)
            ransac_threshold: RANSAC reprojection threshold in pixels
            output_dir: Directory for saving aligned images
        """
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) is required for image registration. "
                            "Install with: pip install opencv-python")
        
        self.detector_type = detector
        self.max_features = max_features
        self.match_ratio = match_ratio
        self.ransac_threshold = ransac_threshold
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_detector(self):
        """Create feature detector based on configuration."""
        if self.detector_type == self.DETECTOR_SIFT:
            return cv2.SIFT_create(nfeatures=self.max_features)
        else:
            return cv2.ORB_create(nfeatures=self.max_features)
    
    def _create_matcher(self):
        """Create feature matcher appropriate for the detector."""
        if self.detector_type == self.DETECTOR_SIFT:
            return cv2.BFMatcher(cv2.NORM_L2)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING)
    
    def _load_grayscale(self, path: str) -> np.ndarray:
        """Load an image as grayscale numpy array."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img
    
    def _load_color(self, path: str) -> np.ndarray:
        """Load an image as color (BGR) numpy array."""
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img
    
    def _upscale_to_match(self, target: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Upscale the target image to approximately match the reference resolution.
        
        Returns:
            (upscaled_image, scale_factor)
        """
        ref_h, ref_w = reference.shape[:2]
        tgt_h, tgt_w = target.shape[:2]
        
        # Calculate scale factor based on height (more stable than width for spreads)
        scale = ref_h / tgt_h
        
        if abs(scale - 1.0) < 0.05:
            # Within 5% — no scaling needed
            return target, 1.0
        
        new_w = int(tgt_w * scale)
        new_h = int(tgt_h * scale)
        
        logger.info(f"Upscaling target from {tgt_w}x{tgt_h} to {new_w}x{new_h} "
                    f"(scale={scale:.2f})")
        
        # Use LANCZOS for upscaling, AREA for downscaling
        interpolation = cv2.INTER_LANCZOS4 if scale > 1.0 else cv2.INTER_AREA
        upscaled = cv2.resize(target, (new_w, new_h), interpolation=interpolation)
        
        return upscaled, scale
    
    def register_pair(self, reference_path: str, target_path: str,
                      save_aligned: bool = True,
                      output_filename: str = None) -> RegistrationResult:
        """
        Register (align) a target image to a reference image.
        
        The target is warped via homography to match the reference geometry.
        If the images have different resolutions, the target is first upscaled.
        
        Args:
            reference_path: Path to the reference (higher-res) page image
            target_path: Path to the target page image to align
            save_aligned: If True, save aligned result to output_dir
            output_filename: Custom filename for the aligned output
            
        Returns:
            RegistrationResult with alignment metrics and aligned image
        """
        result = RegistrationResult(
            reference_path=str(reference_path),
            target_path=str(target_path),
        )
        
        # Load images
        ref_gray = self._load_grayscale(reference_path)
        tgt_gray = self._load_grayscale(target_path)
        tgt_color = self._load_color(target_path)
        
        result.reference_size = (ref_gray.shape[1], ref_gray.shape[0])
        result.target_original_size = (tgt_gray.shape[1], tgt_gray.shape[0])
        
        # Upscale target if resolution mismatch
        tgt_gray_scaled, scale = self._upscale_to_match(tgt_gray, ref_gray)
        tgt_color_scaled, _ = self._upscale_to_match(tgt_color, ref_gray)
        result.scale_factor = scale
        
        # Detect features
        detector = self._create_detector()
        kp_ref, desc_ref = detector.detectAndCompute(ref_gray, None)
        kp_tgt, desc_tgt = detector.detectAndCompute(tgt_gray_scaled, None)
        
        result.num_keypoints_ref = len(kp_ref)
        result.num_keypoints_target = len(kp_tgt)
        
        logger.info(f"Keypoints: reference={len(kp_ref)}, target={len(kp_tgt)}")
        
        if desc_ref is None or desc_tgt is None or len(kp_ref) < 4 or len(kp_tgt) < 4:
            logger.error("Not enough keypoints for registration")
            return result
        
        # Match features
        matcher = self._create_matcher()
        raw_matches = matcher.knnMatch(desc_tgt, desc_ref, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for m_pair in raw_matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < self.match_ratio * n.distance:
                    good_matches.append(m)
        
        result.num_matches = len(good_matches)
        logger.info(f"Good matches after ratio test: {len(good_matches)}")
        
        if len(good_matches) < 4:
            logger.error("Not enough good matches for homography")
            return result
        
        # Compute homography with RANSAC
        src_pts = np.float32([kp_tgt[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        if H is None:
            logger.error("Homography computation failed")
            return result
        
        result.homography = H
        inlier_mask = mask.ravel().astype(bool)
        result.num_inliers = int(inlier_mask.sum())
        result.inlier_ratio = result.num_inliers / len(good_matches) if good_matches else 0
        
        # Compute reprojection error on inliers
        if result.num_inliers > 0:
            src_inliers = src_pts[inlier_mask]
            dst_inliers = dst_pts[inlier_mask]
            projected = cv2.perspectiveTransform(src_inliers, H)
            errors = np.sqrt(np.sum((projected - dst_inliers) ** 2, axis=2))
            result.reprojection_error = float(np.mean(errors))
        
        logger.info(f"Inliers: {result.num_inliers}/{len(good_matches)} "
                    f"({result.inlier_ratio:.1%}), "
                    f"reproj error: {result.reprojection_error:.2f}px, "
                    f"quality: {result.quality_label}")
        
        # Warp the color target to match reference geometry
        ref_h, ref_w = ref_gray.shape[:2]
        aligned = cv2.warpPerspective(tgt_color_scaled, H, (ref_w, ref_h),
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(255, 255, 255))
        
        result.target_aligned_size = (aligned.shape[1], aligned.shape[0])
        
        # Convert to PIL for interoperability
        if PIL_AVAILABLE:
            aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            result.aligned_target = Image.fromarray(aligned_rgb)
        
        # Save if requested
        if save_aligned and self.output_dir:
            if output_filename is None:
                ref_stem = Path(reference_path).stem
                tgt_stem = Path(target_path).stem
                output_filename = f"aligned_{tgt_stem}_to_{ref_stem}.jpg"
            
            output_path = self.output_dir / output_filename
            cv2.imwrite(str(output_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            result.aligned_target_path = str(output_path)
            logger.info(f"Saved aligned image to {output_path}")
        
        return result
    
    def register_page_set(self, reference_dir: str, target_dir: str,
                           page_mapping: Optional[Dict[str, str]] = None) -> List[RegistrationResult]:
        """
        Register all matching pages from two source directories.
        
        Args:
            reference_dir: Directory with reference page images
            target_dir: Directory with target page images
            page_mapping: Optional explicit mapping {ref_filename: target_filename}.
                          If None, matches pages by sort-order prefix (e.g., '010_*').
                          
        Returns:
            List of RegistrationResult for each aligned page pair
        """
        ref_dir = Path(reference_dir)
        tgt_dir = Path(target_dir)
        
        if page_mapping is None:
            page_mapping = self._auto_match_pages(ref_dir, tgt_dir)
        
        logger.info(f"Registering {len(page_mapping)} page pairs")
        
        results = []
        for ref_name, tgt_name in sorted(page_mapping.items()):
            ref_path = ref_dir / ref_name
            tgt_path = tgt_dir / tgt_name
            
            if not ref_path.exists() or not tgt_path.exists():
                logger.warning(f"Skipping missing pair: {ref_name} / {tgt_name}")
                continue
            
            try:
                result = self.register_pair(str(ref_path), str(tgt_path))
                results.append(result)
                
                quality = result.quality_label
                logger.info(f"  {ref_name} ↔ {tgt_name}: {quality} "
                          f"({result.num_inliers} inliers)")
            except Exception as e:
                logger.error(f"Failed to register {ref_name} ↔ {tgt_name}: {e}")
        
        # Summary
        good = sum(1 for r in results if r.is_good_alignment)
        logger.info(f"Registration complete: {good}/{len(results)} pages aligned successfully")
        
        return results
    
    def _auto_match_pages(self, ref_dir: Path, tgt_dir: Path) -> Dict[str, str]:
        """
        Automatically match pages between directories by sort-order prefix.
        
        Both Wright and Aspley images use format: NNN_description.jpg
        We match on the 3-digit prefix since page numbering corresponds.
        """
        ref_files = {f.name[:3]: f.name for f in sorted(ref_dir.glob("*.jpg"))}
        tgt_files = {f.name[:3]: f.name for f in sorted(tgt_dir.glob("*.jpg"))}
        
        mapping = {}
        for prefix, ref_name in ref_files.items():
            if prefix in tgt_files:
                mapping[ref_name] = tgt_files[prefix]
        
        logger.info(f"Auto-matched {len(mapping)} pages by sort-order prefix")
        return mapping
    
    def compute_ssim_score(self, reference_path: str, aligned_path: str,
                           region: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        Compute Structural Similarity Index between reference and aligned images.
        
        Uses a simplified SSIM computation (no scikit-image dependency).
        
        Args:
            reference_path: Path to reference image
            aligned_path: Path to aligned target image
            region: Optional (x, y, w, h) to compute SSIM on a sub-region
            
        Returns:
            SSIM score (0.0 to 1.0, higher = more similar)
        """
        ref = self._load_grayscale(reference_path)
        aligned = self._load_grayscale(aligned_path)
        
        if region:
            x, y, w, h = region
            ref = ref[y:y+h, x:x+w]
            aligned = aligned[y:y+h, x:x+w]
        
        # Ensure same size
        if ref.shape != aligned.shape:
            aligned = cv2.resize(aligned, (ref.shape[1], ref.shape[0]))
        
        # Simplified SSIM (Wang et al. 2004)
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        ref_f = ref.astype(np.float64)
        aligned_f = aligned.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(ref_f, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(aligned_f, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(ref_f ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(aligned_f ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(ref_f * aligned_f, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return float(np.mean(ssim_map))
