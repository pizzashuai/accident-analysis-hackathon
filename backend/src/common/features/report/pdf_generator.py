"""
PDF Report Generation Module

Generates professional PDF reports from LLM analysis results, including
screenshots and map overlays.
"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import base64

from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

logger = logging.getLogger(__name__)


def generate_pdf_report(
    analysis_text: str,
    project_title: str,
    project_description: Optional[str],
    video_screenshot_path: Optional[str],
    map_overlay_path: Optional[str],
    metadata: Dict[str, Any],
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a professional PDF report from LLM analysis results.
    
    Args:
        analysis_text: The LLM analysis report text (markdown format)
        project_title: Project title
        project_description: Project description
        video_screenshot_path: Path to video screenshot image
        map_overlay_path: Path to map overlay image
        metadata: Additional metadata (timestamps, track IDs, etc.)
        output_path: Optional output path for PDF file
        
    Returns:
        Dictionary with generation results:
        - success: bool
        - output_path: str
        - error: str (if failed)
    """
    try:
        # Create temporary file if no output path provided
        if not output_path:
            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            output_path = temp_file.name
            temp_file.close()
        
        output_path = Path(output_path)
        
        # Generate HTML content
        html_content = _generate_html_content(
            analysis_text=analysis_text,
            project_title=project_title,
            project_description=project_description,
            video_screenshot_path=video_screenshot_path,
            map_overlay_path=map_overlay_path,
            metadata=metadata
        )
        
        # Generate CSS styles
        css_content = _generate_css_styles()
        
        # Create HTML document
        html_doc = HTML(string=html_content)
        
        # Create CSS document
        css_doc = CSS(string=css_content)
        
        # Generate PDF
        html_doc.write_pdf(str(output_path), stylesheets=[css_doc])
        
        logger.info(f"Successfully generated PDF report: {output_path}")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        return {
            "success": False,
            "output_path": str(output_path) if output_path else "",
            "error": str(e)
        }


def _generate_html_content(
    analysis_text: str,
    project_title: str,
    project_description: Optional[str],
    video_screenshot_path: Optional[str],
    map_overlay_path: Optional[str],
    metadata: Dict[str, Any]
) -> str:
    """Generate HTML content for the PDF report."""
    
    # Convert images to base64 for embedding
    video_img_data = ""
    map_img_data = ""
    
    if video_screenshot_path and Path(video_screenshot_path).exists():
        try:
            with open(video_screenshot_path, "rb") as f:
                video_img_data = base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.warning(f"Could not encode video screenshot: {e}")
    
    if map_overlay_path and Path(map_overlay_path).exists():
        try:
            with open(map_overlay_path, "rb") as f:
                map_img_data = base64.b64encode(f.read()).decode()
        except Exception as e:
            logger.warning(f"Could not encode map overlay: {e}")
    
    # Format analysis text (convert markdown-like formatting to HTML)
    formatted_analysis = _format_analysis_text(analysis_text)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Accident Analysis Report - {project_title}</title>
    </head>
    <body>
        <div class="header">
            <h1>Accident Analysis Report</h1>
            <h2>{project_title}</h2>
            {f'<p class="description">{project_description}</p>' if project_description else ''}
            <div class="metadata">
                <p><strong>Generated:</strong> {timestamp}</p>
                {f'<p><strong>Analysis ID:</strong> {metadata.get("analysis_id", "N/A")}</p>' if metadata.get("analysis_id") else ''}
                {f'<p><strong>Track IDs:</strong> {", ".join(map(str, metadata.get("track_ids", [])))}</p>' if metadata.get("track_ids") else ''}
            </div>
        </div>
        
        <div class="content">
            <div class="analysis-section">
                <h3>Analysis Results</h3>
                <div class="analysis-content">
                    {formatted_analysis}
                </div>
            </div>
            
            {_generate_screenshots_section(video_img_data, map_img_data, metadata)}
            
            <div class="footer">
                <p>Report generated by Accident Analysis System</p>
                <p>This report contains automated analysis results and should be reviewed by qualified professionals.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


def _format_analysis_text(analysis_text: str) -> str:
    """Convert markdown-like formatting to HTML."""
    # Convert markdown headers to HTML
    formatted = analysis_text.replace("**", "<strong>").replace("**", "</strong>")
    formatted = formatted.replace("*", "<em>").replace("*", "</em>")
    
    # Convert line breaks to paragraphs
    paragraphs = formatted.split('\n\n')
    html_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            # Check if it's a header (starts with #)
            if para.startswith('#'):
                level = len(para) - len(para.lstrip('#'))
                text = para.lstrip('# ').strip()
                html_paragraphs.append(f'<h{level}>{text}</h{level}>')
            else:
                # Convert single line breaks to <br>
                para = para.replace('\n', '<br>')
                html_paragraphs.append(f'<p>{para}</p>')
    
    return '\n'.join(html_paragraphs)


def _generate_screenshots_section(video_img_data: str, map_img_data: str, metadata: Dict[str, Any]) -> str:
    """Generate HTML for screenshots section."""
    if not video_img_data and not map_img_data:
        return ""
    
    screenshots_html = '<div class="screenshots-section"><h3>Visual Analysis</h3><div class="screenshots-grid">'
    
    if video_img_data:
        collision_frame = metadata.get("collision_frame", "N/A")
        collision_timestamp = metadata.get("collision_timestamp", "N/A")
        screenshots_html += f'''
        <div class="screenshot-item">
            <h4>Video Screenshot</h4>
            <img src="data:image/png;base64,{video_img_data}" alt="Video Screenshot" />
            <p class="screenshot-caption">Frame: {collision_frame} | Timestamp: {collision_timestamp}s</p>
        </div>
        '''
    
    if map_img_data:
        collision_point = metadata.get("collision_point", "N/A")
        screenshots_html += f'''
        <div class="screenshot-item">
            <h4>Map Overlay</h4>
            <img src="data:image/png;base64,{map_img_data}" alt="Map Overlay" />
            <p class="screenshot-caption">Collision Point: {collision_point}</p>
        </div>
        '''
    
    screenshots_html += '</div></div>'
    return screenshots_html


def _generate_css_styles() -> str:
    """Generate CSS styles for the PDF report."""
    return """
    @page {
        size: A4;
        margin: 1in;
        @top-center {
            content: "Accident Analysis Report";
            font-size: 10pt;
            color: #666;
        }
        @bottom-right {
            content: "Page " counter(page) " of " counter(pages);
            font-size: 10pt;
            color: #666;
        }
    }
    
    body {
        font-family: 'Arial', sans-serif;
        line-height: 1.6;
        color: #333;
        margin: 0;
        padding: 0;
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    .header h1 {
        color: #2c3e50;
        font-size: 28pt;
        margin: 0 0 0.5rem 0;
        font-weight: bold;
    }
    
    .header h2 {
        color: #34495e;
        font-size: 20pt;
        margin: 0 0 1rem 0;
        font-weight: normal;
    }
    
    .description {
        font-style: italic;
        color: #666;
        margin: 0 0 1rem 0;
    }
    
    .metadata {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
    
    .metadata p {
        margin: 0.25rem 0;
        font-size: 11pt;
    }
    
    .content {
        margin-top: 2rem;
    }
    
    .analysis-section {
        margin-bottom: 2rem;
    }
    
    .analysis-section h3 {
        color: #2c3e50;
        font-size: 18pt;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    .analysis-content {
        font-size: 11pt;
        line-height: 1.7;
    }
    
    .analysis-content h1, .analysis-content h2, .analysis-content h3, 
    .analysis-content h4, .analysis-content h5, .analysis-content h6 {
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .analysis-content h1 { font-size: 16pt; }
    .analysis-content h2 { font-size: 14pt; }
    .analysis-content h3 { font-size: 13pt; }
    .analysis-content h4 { font-size: 12pt; }
    
    .analysis-content p {
        margin-bottom: 1rem;
    }
    
    .analysis-content strong {
        color: #2c3e50;
        font-weight: bold;
    }
    
    .analysis-content em {
        font-style: italic;
        color: #666;
    }
    
    .screenshots-section {
        margin-top: 2rem;
        page-break-inside: avoid;
    }
    
    .screenshots-section h3 {
        color: #2c3e50;
        font-size: 18pt;
        margin-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
    }
    
    .screenshots-grid {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .screenshot-item {
        flex: 1;
        min-width: 300px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .screenshot-item h4 {
        color: #34495e;
        font-size: 14pt;
        margin-bottom: 0.5rem;
    }
    
    .screenshot-item img {
        max-width: 100%;
        height: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .screenshot-caption {
        font-size: 10pt;
        color: #666;
        margin-top: 0.5rem;
        font-style: italic;
    }
    
    .footer {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
        text-align: center;
        font-size: 10pt;
        color: #666;
    }
    
    .footer p {
        margin: 0.25rem 0;
    }
    """
