import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from src.common.database.models.homography_pair_table import HomographyPair


@dataclass
class HomographyResult:
    """Result of homography calculation"""
    matrix: List[List[float]]
    reprojection_error: float
    inlier_count: int
    status: str


def solve_homography_from_pairs(pairs: List[HomographyPair]) -> HomographyResult:
    """
    Calculate homography matrix from point pairs using OpenCV RANSAC.
    
    Args:
        pairs: List of HomographyPair objects from database
        
    Returns:
        HomographyResult with matrix, error, and metadata
        
    Raises:
        ValueError: If insufficient points or calculation fails
    """
    if len(pairs) < 4:
        raise ValueError("At least 4 point pairs are required for homography calculation")
    
    # Extract source points (image coordinates) and destination points (geo coordinates)
    src_points = []
    dst_points = []
    
    for pair in pairs:
        src_points.append([pair.image_x_norm, pair.image_y_norm])
        dst_points.append([pair.map_lng, pair.map_lat])  # Note: lng, lat order for x, y mapping
    
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # Calculate homography matrix using RANSAC
    homography_matrix, mask = cv2.findHomography(
        src_points, dst_points, cv2.RANSAC, 5.0
    )
    
    if homography_matrix is None:
        raise ValueError("Failed to calculate homography matrix")
    
    # Calculate reprojection error
    reprojection_error = _calculate_reprojection_error(
        src_points, dst_points, homography_matrix, mask
    )
    
    # Count inliers
    inlier_count = int(np.sum(mask)) if mask is not None else len(pairs)
    
    # Convert matrix to nested list for JSON serialization
    matrix_list = homography_matrix.tolist()
    
    return HomographyResult(
        matrix=matrix_list,
        reprojection_error=reprojection_error,
        inlier_count=inlier_count,
        status="success"
    )


def _calculate_reprojection_error(
    src_points: np.ndarray, 
    dst_points: np.ndarray, 
    homography_matrix: np.ndarray,
    mask: np.ndarray
) -> float:
    """
    Calculate mean reprojection error for inlier points.
    
    Args:
        src_points: Source points (image coordinates)
        dst_points: Destination points (geo coordinates)  
        homography_matrix: 3x3 homography matrix
        mask: RANSAC mask indicating inliers
        
    Returns:
        Mean reprojection error in coordinate units
    """
    if mask is None:
        mask = np.ones(len(src_points), dtype=bool)
    
    # Transform source points using homography
    src_homogeneous = np.hstack([src_points, np.ones((len(src_points), 1))])
    transformed = src_homogeneous @ homography_matrix.T
    
    # Convert back to 2D coordinates
    transformed_2d = transformed[:, :2] / transformed[:, 2:3]
    
    # Calculate errors for inlier points only
    inlier_indices = np.where(mask)[0]
    if len(inlier_indices) == 0:
        return float('inf')
    
    errors = np.linalg.norm(transformed_2d[inlier_indices] - dst_points[inlier_indices], axis=1)
    mean_error = np.mean(errors)
    
    return float(mean_error)


def validate_homography_matrix(matrix: List[List[float]]) -> bool:
    """
    Validate that a homography matrix is well-formed.
    
    Args:
        matrix: 3x3 matrix as nested list
        
    Returns:
        True if matrix is valid, False otherwise
    """
    try:
        matrix_np = np.array(matrix, dtype=np.float32)
        
        # Check shape
        if matrix_np.shape != (3, 3):
            return False
            
        # Check that bottom row is [0, 0, 1] (homogeneous coordinates)
        if not np.allclose(matrix_np[2, :], [0, 0, 1]):
            return False
            
        # Check determinant is non-zero
        if abs(np.linalg.det(matrix_np)) < 1e-10:
            return False
            
        return True
        
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return False


def transform_point(
    x_norm: float, 
    y_norm: float, 
    matrix: List[List[float]]
) -> Tuple[float, float]:
    """
    Transform a normalized image point to geographic coordinates.
    
    Args:
        x_norm: Normalized x coordinate (0-1)
        y_norm: Normalized y coordinate (0-1)
        matrix: 3x3 homography matrix
        
    Returns:
        Tuple of (longitude, latitude)
    """
    matrix_np = np.array(matrix, dtype=np.float32)
    
    # Create point in homogeneous coordinates
    point = np.array([[x_norm, y_norm]], dtype=np.float32)
    
    # Transform using homography
    transformed = cv2.perspectiveTransform(
        point.reshape(-1, 1, 2), matrix_np
    )
    
    lng, lat = transformed[0][0]
    return float(lng), float(lat)
