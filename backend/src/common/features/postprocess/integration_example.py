"""
Integration examples showing how to use the accident analysis agent
from different parts of the backend (API routes, Celery tasks, etc.)
"""

from typing import Dict, Any, List
from pathlib import Path
from agent_core import AccidentAnalysisAgent, AgentConfig


# Example 1: Simple function call from API route
def analyze_collision_simple(
    track_ids: List[int],
    detections_file: str
) -> Dict[str, Any]:
    """
    Simple collision analysis that can be called from an API route.
    
    Args:
        track_ids: List of track IDs to analyze (e.g., [7, 14])
        detections_file: Path to detections.jsonl file
        
    Returns:
        Analysis report dictionary
    """
    config = AgentConfig(
        track_ids=track_ids,
        detections_file=detections_file
    )
    
    agent = AccidentAnalysisAgent(config)
    report = agent.analyze()
    
    return report


# Example 2: Async-compatible wrapper for FastAPI
async def analyze_collision_async(
    track_ids: List[int],
    detections_file: str,
    frame_range: tuple = None,
    iou_threshold: float = 0.01,
    distance_threshold_m: float = 5.0
) -> Dict[str, Any]:
    """
    Async wrapper for collision analysis (FastAPI-compatible).
    
    Args:
        track_ids: Track IDs to analyze
        detections_file: Path to detections file
        frame_range: Optional frame range tuple
        iou_threshold: IoU threshold
        distance_threshold_m: Distance threshold in meters
        
    Returns:
        Analysis report
    """
    import asyncio
    
    def _analyze():
        config = AgentConfig(
            track_ids=track_ids,
            frame_range=frame_range,
            iou_threshold=iou_threshold,
            distance_threshold_m=distance_threshold_m,
            detections_file=detections_file
        )
        
        agent = AccidentAnalysisAgent(config)
        return agent.analyze()
    
    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _analyze)


# Example 3: Celery task wrapper
def create_celery_task():
    """
    Example Celery task for background collision analysis.
    Add this to your Celery worker tasks.
    """
    from celery import shared_task
    
    @shared_task(name="analyze_collision")
    def analyze_collision_task(
        project_id: int,
        track_ids: List[int],
        detections_file: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Celery task for background collision analysis.
        
        Args:
            project_id: Project ID for tracking
            track_ids: Track IDs to analyze
            detections_file: Path to detections file
            **kwargs: Additional config options
            
        Returns:
            Analysis report
        """
        try:
            config = AgentConfig(
                track_ids=track_ids,
                detections_file=detections_file,
                frame_range=kwargs.get('frame_range'),
                iou_threshold=kwargs.get('iou_threshold', 0.01),
                distance_threshold_m=kwargs.get('distance_threshold_m', 5.0),
                persistence_frames=kwargs.get('persistence_frames', 3)
            )
            
            agent = AccidentAnalysisAgent(config)
            report = agent.analyze()
            
            # You could save report to database here
            # save_analysis_report(project_id, report)
            
            return {
                "success": True,
                "project_id": project_id,
                "report": report
            }
            
        except Exception as e:
            return {
                "success": False,
                "project_id": project_id,
                "error": str(e)
            }
    
    return analyze_collision_task


# Example 4: FastAPI route handler
def create_fastapi_endpoint():
    """
    Example FastAPI endpoint for collision analysis.
    Add this to your API routes.
    """
    from fastapi import APIRouter, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    
    router = APIRouter(prefix="/analysis", tags=["analysis"])
    
    class AnalysisRequest(BaseModel):
        project_id: int
        track_ids: List[int]
        frame_range: tuple | None = None
        iou_threshold: float = 0.01
        distance_threshold_m: float = 5.0
    
    class AnalysisResponse(BaseModel):
        success: bool
        collision_detected: bool
        narrative_summary: str
        timeline: List[str]
        impact_analysis: Dict[str, Any]
        data_quality: Dict[str, Any]
    
    @router.post("/collision", response_model=AnalysisResponse)
    async def analyze_collision(request: AnalysisRequest):
        """
        Analyze collision between two vehicles.
        
        Expects a detections.jsonl file to be available for the project.
        """
        # Construct path to detections file
        detections_file = f"uploads/{request.project_id}/detections.jsonl"
        
        if not Path(detections_file).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Detections file not found for project {request.project_id}"
            )
        
        # Run analysis
        report = await analyze_collision_async(
            track_ids=request.track_ids,
            detections_file=detections_file,
            frame_range=request.frame_range,
            iou_threshold=request.iou_threshold,
            distance_threshold_m=request.distance_threshold_m
        )
        
        if not report.get('success'):
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {report.get('error', 'Unknown error')}"
            )
        
        return AnalysisResponse(**report)
    
    @router.post("/collision/background")
    async def analyze_collision_background(
        request: AnalysisRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Start collision analysis as a background task.
        Returns immediately with a task ID.
        """
        # Queue analysis task
        task_id = "task_123"  # Generate unique task ID
        
        # Add to background tasks or Celery
        # task = analyze_collision_task.apply_async(
        #     args=[request.project_id, request.track_ids, detections_file],
        #     kwargs={"iou_threshold": request.iou_threshold}
        # )
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Analysis started in background"
        }
    
    return router


# Example 5: Database integration
class AnalysisResult:
    """Example SQLAlchemy model for storing analysis results."""
    
    @staticmethod
    def save_to_database(project_id: int, report: Dict[str, Any], db_session):
        """
        Save analysis report to database.
        
        Args:
            project_id: Project ID
            report: Analysis report from agent
            db_session: SQLAlchemy session
        """
        # This is a pseudo-code example
        # Adjust based on your actual database models
        
        from datetime import datetime
        
        analysis_record = {
            "project_id": project_id,
            "collision_detected": report['collision_detected'],
            "narrative_summary": report['narrative_summary'],
            "timeline_json": report['timeline'],
            "impact_analysis_json": report['impact_analysis'],
            "data_quality_json": report['data_quality'],
            "metrics_summary_json": report['metrics_summary'],
            "created_at": datetime.utcnow()
        }
        
        # Insert into database
        # db_session.execute(
        #     "INSERT INTO collision_analysis (...) VALUES (...)",
        #     analysis_record
        # )
        # db_session.commit()
        
        return analysis_record


# Example 6: Batch analysis
def analyze_multiple_collisions(
    collision_pairs: List[tuple],
    detections_file: str
) -> List[Dict[str, Any]]:
    """
    Analyze multiple collision pairs in batch.
    
    Args:
        collision_pairs: List of track ID pairs, e.g., [(7, 14), (3, 8), ...]
        detections_file: Path to detections file
        
    Returns:
        List of analysis reports
    """
    reports = []
    
    for track_ids in collision_pairs:
        try:
            config = AgentConfig(
                track_ids=list(track_ids),
                detections_file=detections_file
            )
            
            agent = AccidentAnalysisAgent(config)
            report = agent.analyze()
            reports.append(report)
            
        except Exception as e:
            reports.append({
                "success": False,
                "track_ids": track_ids,
                "error": str(e)
            })
    
    return reports


# Example 7: Streamlined API response
def format_for_frontend(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format analysis report for frontend consumption.
    Simplifies the structure and extracts key information.
    
    Args:
        report: Full analysis report from agent
        
    Returns:
        Simplified report for frontend
    """
    if not report.get('success'):
        return {
            "error": report.get('error', 'Analysis failed'),
            "collision_detected": False
        }
    
    collision = report['collision_detected']
    impact = report['impact_analysis']
    
    result = {
        "collision_detected": collision,
        "summary": report['narrative_summary'],
        "timeline": [
            {
                "stage": entry.split(']')[0].replace('[', '').strip(),
                "description": entry.split(']')[1].strip() if ']' in entry else entry
            }
            for entry in report['timeline']
        ],
        "key_metrics": {
            "max_iou": report['metrics_summary']['max_iou'],
            "min_distance_m": report['metrics_summary']['min_distance_m'],
            "frames_analyzed": report['data_quality']['total_frames_analyzed']
        }
    }
    
    if collision:
        result["impact"] = {
            "first_contact_frame": impact['key_frames']['first_contact'],
            "duration_frames": impact['severity_indicators']['impact_duration_frames'],
            "severity": "high" if impact['severity_indicators']['max_iou'] > 0.05 else "moderate"
        }
    else:
        result["near_miss"] = {
            "closest_distance_m": impact['closest_approach']['distance_m'],
            "closest_frame": impact['closest_approach']['frame']
        }
    
    # Include data quality warnings
    result["warnings"] = report['data_quality']['assumptions']
    
    return result


# Example usage in a real application
if __name__ == "__main__":
    # Example: Analyze from project uploads
    project_id = 1
    track_ids = [7, 14]
    detections_file = "../process_video/detections.jsonl"
    
    # Run analysis
    report = analyze_collision_simple(track_ids, detections_file)
    
    # Format for frontend
    frontend_response = format_for_frontend(report)
    
    print("Frontend Response:")
    import json
    print(json.dumps(frontend_response, indent=2))
    
    # Example: Batch analysis
    collision_pairs = [(7, 14), (1, 2), (3, 5)]
    batch_reports = analyze_multiple_collisions(collision_pairs, detections_file)
    
    print(f"\nBatch Analysis: Analyzed {len(batch_reports)} collision pairs")
    for i, report in enumerate(batch_reports, 1):
        if report.get('success'):
            print(f"  {i}. Tracks {collision_pairs[i-1]}: {'COLLISION' if report['collision_detected'] else 'NO COLLISION'}")
        else:
            print(f"  {i}. Tracks {collision_pairs[i-1]}: ERROR - {report.get('error')}")

