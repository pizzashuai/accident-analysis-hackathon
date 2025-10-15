import json
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple
from math import radians, cos, sin, asin, sqrt


@dataclass
class ImagePoint:
    """Represents a point in image coordinates (normalized)"""

    xNorm: float
    yNorm: float


@dataclass
class GeoPoint:
    """Represents a geographic point (latitude, longitude)"""

    lat: float
    lng: float


@dataclass
class PointPair:
    """Represents a pair of corresponding points: image point (a) and geo point (b)"""

    id: int
    a: ImagePoint
    b: GeoPoint


@dataclass
class HomographyData:
    """Stores all homography point pairs and metadata"""

    pairs: List[PointPair]
    imagesMeta: dict
    mapMeta: dict


class DistanceEstimator:
    """Estimates real-world distances between image points using homography"""

    def __init__(self, homography_file: str):
        """
        Initialize the distance estimator with homography data.

        Args:
            homography_file: Path to JSON file containing point pairs
        """
        self.homography_data = self._load_homography_data(homography_file)
        self.homography_matrix = self._calculate_homography()

    def _load_homography_data(self, file_path: str) -> HomographyData:
        """Load homography point pairs from JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)

        pairs = []
        for pair_data in data["pairs"]:
            image_point = ImagePoint(
                xNorm=pair_data["a"]["xNorm"], yNorm=pair_data["a"]["yNorm"]
            )
            geo_point = GeoPoint(lat=pair_data["b"]["lat"], lng=pair_data["b"]["lng"])
            pairs.append(PointPair(id=pair_data["id"], a=image_point, b=geo_point))

        return HomographyData(
            pairs=pairs,
            imagesMeta=data.get("imagesMeta", {}),
            mapMeta=data.get("mapMeta", {}),
        )

    def _calculate_homography(self) -> np.ndarray:
        """
        Calculate homography matrix from image coordinates to geo coordinates.

        Returns:
            3x3 homography matrix
        """
        # Extract source points (image coordinates) and destination points (geo coordinates)
        src_points = []
        dst_points = []

        for pair in self.homography_data.pairs:
            src_points.append([pair.a.xNorm, pair.a.yNorm])
            dst_points.append(
                [pair.b.lng, pair.b.lat]
            )  # Note: lng, lat order for x, y mapping

        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Calculate homography matrix
        # We need at least 4 point pairs for homography
        if len(src_points) < 4:
            raise ValueError(
                "At least 4 point pairs are required for homography calculation"
            )

        homography_matrix, status = cv2.findHomography(
            src_points, dst_points, cv2.RANSAC, 5.0
        )

        if homography_matrix is None:
            raise ValueError("Failed to calculate homography matrix")

        return homography_matrix

    def image_to_geo(self, x_norm: float, y_norm: float) -> GeoPoint:
        """
        Transform image coordinates to geographic coordinates using homography.

        Args:
            x_norm: Normalized x coordinate in image (0-1)
            y_norm: Normalized y coordinate in image (0-1)

        Returns:
            GeoPoint with lat/lng coordinates
        """
        # Create point in homogeneous coordinates
        point = np.array([[x_norm, y_norm]], dtype=np.float32)

        # Transform using homography
        transformed = cv2.perspectiveTransform(
            point.reshape(-1, 1, 2), self.homography_matrix
        )

        lng, lat = transformed[0][0]
        return GeoPoint(lat=lat, lng=lng)

    @staticmethod
    def haversine_distance(point1: GeoPoint, point2: GeoPoint) -> float:
        """
        Calculate the great circle distance between two points on Earth.

        Args:
            point1: First geographic point
            point2: Second geographic point

        Returns:
            Distance in meters
        """
        # Radius of Earth in meters
        R = 6371000

        # Convert to radians
        lat1, lng1 = radians(point1.lat), radians(point1.lng)
        lat2, lng2 = radians(point2.lat), radians(point2.lng)

        # Haversine formula
        dlat = lat2 - lat1
        dlng = lng2 - lng1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlng / 2) ** 2
        c = 2 * asin(sqrt(a))

        distance = R * c
        return distance

    def estimate_distance(
        self, point1: Tuple[float, float], point2: Tuple[float, float]
    ) -> float:
        """
        Estimate real-world distance between two image points.

        Args:
            point1: Tuple of (x_norm, y_norm) for first point in image
            point2: Tuple of (x_norm, y_norm) for second point in image

        Returns:
            Distance in meters
        """
        # Transform image points to geo coordinates
        geo_point1 = self.image_to_geo(point1[0], point1[1])
        geo_point2 = self.image_to_geo(point2[0], point2[1])

        # Calculate distance
        distance = self.haversine_distance(geo_point1, geo_point2)

        return distance

    def get_homography_matrix(self) -> np.ndarray:
        """Return the calculated homography matrix"""
        return self.homography_matrix


def main():
    """Test the distance estimator"""
    # Initialize estimator with homography data
    estimator = DistanceEstimator("homography-points.json")

    print("=" * 60)
    print("Distance Estimator - Test")
    print("=" * 60)

    # Print homography matrix
    print("\nHomography Matrix:")
    print(estimator.get_homography_matrix())

    # Test: Transform some known points and verify
    print("\n" + "=" * 60)
    print("Verification of known points:")
    print("=" * 60)
    for pair in estimator.homography_data.pairs:
        transformed = estimator.image_to_geo(pair.a.xNorm, pair.a.yNorm)
        print(f"\nPoint {pair.id}:")
        print(f"  Image coords: ({pair.a.xNorm:.4f}, {pair.a.yNorm:.4f})")
        print(f"  Expected geo: ({pair.b.lat:.6f}, {pair.b.lng:.6f})")
        print(f"  Transformed:  ({transformed.lat:.6f}, {transformed.lng:.6f})")
        lat_error = abs(transformed.lat - pair.b.lat)
        lng_error = abs(transformed.lng - pair.b.lng)
        print(f"  Error: lat={lat_error:.8f}, lng={lng_error:.8f}")

    # Test: Calculate distance between two arbitrary points
    print("\n" + "=" * 60)
    print("Distance Estimation Test:")
    print("=" * 60)

    # Example: Distance between two points in the image
    point1 = (0.3, 0.3)  # normalized image coordinates
    point2 = (0.7, 0.6)  # normalized image coordinates

    geo1 = estimator.image_to_geo(point1[0], point1[1])
    geo2 = estimator.image_to_geo(point2[0], point2[1])

    distance = estimator.estimate_distance(point1, point2)

    print(f"\nPoint 1 (image): {point1}")
    print(f"Point 1 (geo):   ({geo1.lat:.6f}, {geo1.lng:.6f})")
    print(f"\nPoint 2 (image): {point2}")
    print(f"Point 2 (geo):   ({geo2.lat:.6f}, {geo2.lng:.6f})")
    print(
        f"\nEstimated distance: {distance:.2f} meters ({distance * 3.28084:.2f} feet)"
    )

    # Test: Calculate distance between first and last calibration points
    print("\n" + "=" * 60)
    print("Distance between calibration points:")
    print("=" * 60)

    first_pair = estimator.homography_data.pairs[0]
    last_pair = estimator.homography_data.pairs[-1]

    point_a = (first_pair.a.xNorm, first_pair.a.yNorm)
    point_b = (last_pair.a.xNorm, last_pair.a.yNorm)

    distance_ab = estimator.estimate_distance(point_a, point_b)

    print(f"\nFirst calibration point: {point_a}")
    print(f"Last calibration point:  {point_b}")
    print(f"Distance: {distance_ab:.2f} meters ({distance_ab * 3.28084:.2f} feet)")


if __name__ == "__main__":
    main()
