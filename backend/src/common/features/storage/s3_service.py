"""S3 service utilities for file operations."""

import logging
from typing import BinaryIO

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from src.common.config import settings

logger = logging.getLogger(__name__)

# Initialize S3 client
_s3_client = None


def get_s3_client():
    """Get or create S3 client."""
    global _s3_client
    if _s3_client is None:
        try:
            _s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION,
            )
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise ValueError("AWS credentials not configured")
    return _s3_client


def upload_file_to_s3(file_obj: BinaryIO, bucket: str, key: str) -> dict:
    """
    Upload file to S3.
    
    Args:
        file_obj: File-like object to upload
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        dict: Upload result with metadata
        
    Raises:
        ClientError: If S3 upload fails
        ValueError: If credentials are not configured
    """
    try:
        client = get_s3_client()
        
        # Reset file pointer to beginning
        file_obj.seek(0)
        
        # Upload file
        response = client.upload_fileobj(
            file_obj,
            bucket,
            key,
            ExtraArgs={
                'ContentType': 'application/octet-stream',
            }
        )
        
        logger.info(f"Successfully uploaded file to S3: s3://{bucket}/{key}")
        
        return {
            "bucket": bucket,
            "key": key,
            "uri": f"s3://{bucket}/{key}",
            "success": True
        }
        
    except ClientError as e:
        logger.error(f"Failed to upload file to S3: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading to S3: {e}")
        raise


def generate_presigned_url(bucket: str, key: str, expiration: int = None) -> str:
    """
    Generate presigned URL for S3 object.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        expiration: URL expiration time in seconds (defaults to settings value)
        
    Returns:
        str: Presigned URL
        
    Raises:
        ClientError: If presigned URL generation fails
        ValueError: If credentials are not configured
    """
    try:
        client = get_s3_client()
        
        if expiration is None:
            expiration = settings.S3_PRESIGNED_URL_EXPIRATION
            
        # Generate presigned URL
        presigned_url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        
        logger.info(f"Generated presigned URL for s3://{bucket}/{key}")
        
        return presigned_url
        
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating presigned URL: {e}")
        raise


def delete_file_from_s3(bucket: str, key: str) -> bool:
    """
    Delete file from S3.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        bool: True if deletion successful
        
    Raises:
        ClientError: If S3 deletion fails
        ValueError: If credentials are not configured
    """
    try:
        client = get_s3_client()
        
        # Delete object
        client.delete_object(Bucket=bucket, Key=key)
        
        logger.info(f"Successfully deleted file from S3: s3://{bucket}/{key}")
        
        return True
        
    except ClientError as e:
        logger.error(f"Failed to delete file from S3: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting from S3: {e}")
        raise


def download_file_from_s3(bucket: str, key: str, local_path: str) -> bool:
    """
    Download file from S3 to local path.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        local_path: Local file path to save the downloaded file
        
    Returns:
        bool: True if download successful
        
    Raises:
        ClientError: If S3 download fails
        ValueError: If credentials are not configured
    """
    try:
        client = get_s3_client()
        
        # Download file
        client.download_file(bucket, key, local_path)
        
        logger.info(f"Successfully downloaded file from S3: s3://{bucket}/{key} to {local_path}")
        
        return True
        
    except ClientError as e:
        logger.error(f"Failed to download file from S3: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading from S3: {e}")
        raise


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Parse S3 URI to extract bucket and key.
    
    Args:
        uri: S3 URI in format s3://bucket/key
        
    Returns:
        tuple: (bucket, key)
        
    Raises:
        ValueError: If URI format is invalid
    """
    if not uri.startswith('s3://'):
        raise ValueError(f"Invalid S3 URI format: {uri}")
    
    # Remove s3:// prefix
    path = uri[5:]
    
    # Split into bucket and key
    parts = path.split('/', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 URI format: {uri}")
    
    bucket, key = parts
    return bucket, key
