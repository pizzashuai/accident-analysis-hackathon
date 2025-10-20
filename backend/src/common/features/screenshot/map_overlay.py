"""Google Static Maps API overlay generation."""

import logging
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Union, List, Tuple

import requests

logger = logging.getLogger(__name__)


def generate_map_overlay(
    center_lat: float,
    center_lng: float,
    trajectories: List[Dict[str, Any]],
    collision_point: Tuple[float, float],
    api_key: str,
    output_path: Union[str, Path],
    zoom: int = 19,
    map_size: str = "640x640"
) -> Dict[str, Any]:
    """
    Generate a Google Static Maps overlay with vehicle trajectories and collision point.
    
    Args:
        center_lat: Center latitude for the map
        center_lng: Center longitude for the map
        trajectories: List of trajectory data with 'track_id', 'world_coords', 'timestamps'
        collision_point: Tuple of (lat, lng) for collision marker
        api_key: Google Maps API key
        output_path: Path to save the map image
        zoom: Map zoom level (default 18)
        map_size: Map size in format "WIDTHxHEIGHT" (default "640x640")
        
    Returns:
        Dictionary with generation results:
        - success: bool
        - output_path: str
        - api_url: str
        - error: str (if failed)
    """
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Validate inputs
        if not api_key or api_key.strip() == "":
            return {
                "success": False,
                "output_path": str(output_path),
                "api_url": "",
                "error": "Google Maps API key is required"
            }
        
        # Validate coordinates
        if not (-90 <= center_lat <= 90):
            return {
                "success": False,
                "output_path": str(output_path),
                "api_url": "",
                "error": f"Invalid center latitude: {center_lat}"
            }
        
        if not (-180 <= center_lng <= 180):
            return {
                "success": False,
                "output_path": str(output_path),
                "api_url": "",
                "error": f"Invalid center longitude: {center_lng}"
            }
        
        collision_lat, collision_lng = collision_point
        if not (-90 <= collision_lat <= 90) or not (-180 <= collision_lng <= 180):
            return {
                "success": False,
                "output_path": str(output_path),
                "api_url": "",
                "error": f"Invalid collision point coordinates: {collision_point}"
            }
        
        # Build the Google Static Maps API URL
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        
        params = {
            "center": f"{center_lat},{center_lng}",
            "zoom": str(zoom),
            "size": map_size,
            "key": api_key,
            "format": "png",
            "maptype": "roadmap"
        }
        
        # Add trajectory paths
        path_params = []
        for i, trajectory in enumerate(trajectories):
            if not trajectory.get('world_coords') or not trajectory.get('timestamps'):
                continue
                
            # Create path from trajectory coordinates
            coords = trajectory['world_coords']
            if len(coords) < 2:
                continue
            
            # Limit path points to avoid URL length issues
            max_points = 30  # Reduced for better visibility
            if len(coords) > max_points:
                # Sample points evenly
                step = len(coords) // max_points
                coords = coords[::step]
            
            # Format coordinates as lat,lng|lat,lng|...
            path_coords = "|".join([f"{lat},{lng}" for lat, lng in coords])
            
            # Add path parameter
            color = f"0x{_get_trajectory_color(i):06x}ff"  # Different color for each trajectory
            path_params.append(f"color:{color}|weight:4|{path_coords}")
            
            # Limit number of paths to avoid URL length issues
            if len(path_params) >= 5:  # Reduced limit for better performance
                logger.warning(f"Limited to {len(path_params)} paths due to API limits")
                break
        
        # Add collision marker with explosion symbol
        # Using explosion emoji as label since custom icons can be unreliable
        collision_marker = f"color:red|label:💥|{collision_lat},{collision_lng}"
        
        # Add legend markers in the top-left corner of the map
        # Calculate legend position (offset from map center)
        legend_offset_lat = 0.0005  # Small offset for legend positioning
        legend_offset_lng = -0.0008
        
        legend_markers = []
        
        # Add legend markers for each trajectory
        for i, trajectory in enumerate(trajectories):
            if not trajectory.get('world_coords') or not trajectory.get('timestamps'):
                continue
                
            track_id = trajectory.get('track_id', f'Track {i+1}')
            color = _get_trajectory_color(i)
            color_hex = f"0x{color:06x}"
            
            # Position legend markers in top-left area
            legend_lat = center_lat + legend_offset_lat - (i * 0.0003)
            legend_lng = center_lng + legend_offset_lng
            
            # Create legend marker with track ID label
            legend_marker = f"color:{color_hex}|label:{track_id}|{legend_lat},{legend_lng}"
            legend_markers.append(legend_marker)
        
        # Combine all markers
        all_markers = [collision_marker] + legend_markers
        
        # Build URL with multiple path and marker parameters
        url_parts = [f"{base_url}?"]
        
        # Add basic parameters (excluding path and markers)
        basic_params = {k: v for k, v in params.items() if k not in ["path", "markers"]}
        url_parts.append(urllib.parse.urlencode(basic_params))
        
        # Add path parameters separately
        for path_param in path_params:
            url_parts.append(f"&path={urllib.parse.quote(path_param)}")
        
        # Add marker parameters separately
        for marker in all_markers:
            url_parts.append(f"&markers={urllib.parse.quote(marker)}")
        
        api_url = "".join(url_parts)
        
        # Log the request for debugging (without API key)
        debug_url = api_url.replace(api_key, "***API_KEY***")
        logger.info(f"Making Google Static Maps API request: {debug_url}")
        
        # Make HTTP request
        response = requests.get(api_url, timeout=30)
        
        # Check for API-specific error responses
        if response.status_code != 200:
            error_text = response.text
            logger.error(f"Google Maps API error {response.status_code}: {error_text}")
            return {
                "success": False,
                "output_path": str(output_path),
                "api_url": api_url,
                "error": f"Google Maps API error: {response.status_code} - {error_text}"
            }
        
        # Check if response is actually an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            error_text = response.text
            logger.error(f"Expected image response, got: {content_type} - {error_text}")
            return {
                "success": False,
                "output_path": str(output_path),
                "api_url": api_url,
                "error": f"Expected image response, got: {content_type} - {error_text}"
            }
        
        # Check for API warnings in headers
        warning_header = response.headers.get('X-Staticmap-API-Warning')
        if warning_header:
            logger.warning(f"Google Maps API warning: {warning_header}")
        
        # Save the image
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        logger.info(f"Successfully generated map overlay to {output_path}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "api_url": api_url,
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Error generating map overlay: {e}")
        return {
            "success": False,
            "output_path": str(output_path),
            "api_url": "",
            "error": str(e)
        }


def _get_trajectory_color(index: int) -> int:
    """Get a distinct color for each trajectory."""
    colors = [
        0xff0000,  # Red
        0x0000ff,  # Blue
        0x00ff00,  # Green
        0xffff00,  # Yellow
        0xff00ff,  # Magenta
        0x00ffff,  # Cyan
        0xff8000,  # Orange
        0x8000ff,  # Purple
    ]
    return colors[index % len(colors)]


def calculate_map_center_and_zoom(
    trajectories: List[Dict[str, Any]], 
    collision_point: Tuple[float, float]
) -> Tuple[float, float, int]:
    """
    Calculate optimal map center and zoom level based on trajectory data.
    
    Args:
        trajectories: List of trajectory data
        collision_point: Tuple of (lat, lng) for collision point
        
    Returns:
        Tuple of (center_lat, center_lng, zoom_level)
    """
    all_lats = []
    all_lngs = []
    
    # Collect all coordinates from trajectories
    for trajectory in trajectories:
        coords = trajectory.get('world_coords', [])
        for lat, lng in coords:
            all_lats.append(lat)
            all_lngs.append(lng)
    
    # Add collision point
    collision_lat, collision_lng = collision_point
    all_lats.append(collision_lat)
    all_lngs.append(collision_lng)
    
    if not all_lats or not all_lngs:
        # Fallback to collision point
        return collision_lat, collision_lng, 18
    
    # Calculate center
    center_lat = sum(all_lats) / len(all_lats)
    center_lng = sum(all_lngs) / len(all_lngs)
    
    # Calculate bounding box
    min_lat = min(all_lats)
    max_lat = max(all_lats)
    min_lng = min(all_lngs)
    max_lng = max(all_lngs)
    
    # Calculate zoom level based on bounding box size
    lat_range = max_lat - min_lat
    lng_range = max_lng - min_lng
    max_range = max(lat_range, lng_range)
    
    # Simple zoom calculation (can be refined)
    if max_range > 0.01:  # ~1km
        zoom = 15
    elif max_range > 0.005:  # ~500m
        zoom = 16
    elif max_range > 0.002:  # ~200m
        zoom = 17
    else:
        zoom = 18
    
    return center_lat, center_lng, zoom


def filter_detections_by_timestamp(
    detections: List[Dict[str, Any]], 
    timestamp: float, 
    tolerance: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Filter detections to get those closest to the specified timestamp.
    
    Args:
        detections: List of detection dictionaries
        timestamp: Target timestamp in seconds
        tolerance: Time tolerance in seconds
        
    Returns:
        List of detections within tolerance of the timestamp
    """
    filtered = []
    
    for detection in detections:
        detection_time = detection.get('time', 0)
        if abs(detection_time - timestamp) <= tolerance:
            filtered.append(detection)
    
    return filtered


def extract_trajectories_from_detections(
    detections: List[Dict[str, Any]], 
    track_ids: List[int]
) -> List[Dict[str, Any]]:
    """
    Extract trajectory data for specified track IDs from detections.
    
    Args:
        detections: List of detection dictionaries
        track_ids: List of track IDs to extract
        
    Returns:
        List of trajectory dictionaries with 'track_id', 'world_coords', 'timestamps'
    """
    trajectories = {}
    
    for detection in detections:
        track_id = detection.get('track_id')
        if track_id not in track_ids:
            continue
            
        if track_id not in trajectories:
            trajectories[track_id] = {
                'track_id': track_id,
                'world_coords': [],
                'timestamps': []
            }
        
        world_coords = detection.get('world_coords')
        if world_coords and len(world_coords) == 2:
            # Convert from [lng, lat] to [lat, lng] format
            lng, lat = world_coords
            trajectories[track_id]['world_coords'].append([lat, lng])
            trajectories[track_id]['timestamps'].append(detection.get('time', 0))
    
    # Sort by timestamp for each trajectory
    for trajectory in trajectories.values():
        if trajectory['timestamps']:
            # Sort both lists by timestamp
            sorted_data = sorted(zip(trajectory['timestamps'], trajectory['world_coords']))
            trajectory['timestamps'], trajectory['world_coords'] = zip(*sorted_data)
            trajectory['timestamps'] = list(trajectory['timestamps'])
            trajectory['world_coords'] = list(trajectory['world_coords'])
    
    return list(trajectories.values())
