"""
S3 Utility for uploading interview recordings to AWS S3
"""
import boto3
import os
from datetime import datetime
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger("s3-utils")

# S3 Configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "edy-temp-videos")

# Cached S3 client
_s3_client = None

def get_s3_client():
    """Get or create cached S3 client"""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    return _s3_client

def upload_to_s3(local_file_path: str, session_id: str, candidate_uuid: str, file_type: str) -> str:
    """
    Upload a file to S3 from local disk and return its URL
    """
    try:
        s3_client = get_s3_client()
        filename = os.path.basename(local_file_path)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"ai_interview_recordings/{session_id}/{file_type}/{candidate_uuid}_{timestamp}_{filename}"
        
        s3_client.upload_file(local_file_path, AWS_S3_BUCKET, s3_key)
        s3_url = f"s3://{AWS_S3_BUCKET}/{s3_key}"
        logger.info(f"✅ Uploaded to S3: {s3_key}")
        return s3_url
    except Exception as e:
        logger.error(f"❌ S3 upload failed: {e}")
        return None


def upload_bytes_to_s3(file_bytes: bytes, filename: str, session_id: str, candidate_uuid: str, file_type: str) -> str:
    """
    Upload file bytes DIRECTLY to S3 (no local disk needed)
    
    Args:
        file_bytes: Raw file content as bytes
        filename: Original filename
        session_id: Interview session UUID
        candidate_uuid: Candidate UUID for easy search
        file_type: Type of recording ('audio', 'combined_audio', 'screen', 'transcript')
    
    Returns:
        S3 URL of the uploaded file
    """
    try:
        import io
        s3_client = get_s3_client()
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        s3_key = f"ai_interview_recordings/{session_id}/{file_type}/{candidate_uuid}_{timestamp}_{filename}"
        
        # Upload bytes directly using put_object (no temp file needed)
        s3_client.put_object(
            Bucket=AWS_S3_BUCKET,
            Key=s3_key,
            Body=file_bytes,
            ContentType='video/webm' if filename.endswith('.webm') else 'application/octet-stream'
        )
        
        s3_url = f"s3://{AWS_S3_BUCKET}/{s3_key}"
        logger.info(f"✅ Direct upload to S3: {s3_key} ({len(file_bytes)} bytes)")
        return s3_url
        
    except Exception as e:
        logger.error(f"❌ S3 direct upload failed: {e}")
        return None

def get_presigned_url(s3_url: str, expiration=3600) -> str:
    """
    Generate a presigned URL for private S3 object
    
    Args:
        s3_url: S3 URL (s3://bucket/key format)
        expiration: URL expiration time in seconds (default 1 hour)
    
    Returns:
        Presigned HTTPS URL
    """
    try:
        # Parse S3 URL
        if not s3_url.startswith("s3://"):
            return s3_url
        
        parts = s3_url.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        s3_client = get_s3_client()
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': key},
            ExpiresIn=expiration
        )
        
        return presigned_url
        
    except Exception as e:
        logger.error(f"❌ Failed to generate presigned URL: {e}")
        return None

def list_recordings(session_id: str = None, candidate_uuid: str = None):
    """
    List recordings from S3
    
    Args:
        session_id: Filter by session ID (optional)
        candidate_uuid: Filter by candidate UUID (optional)
    
    Returns:
        List of S3 objects
    """
    try:
        s3_client = get_s3_client()
        
        # Build prefix based on filters
        if session_id:
            prefix = f"ai_interview_recordings/{session_id}/"
        else:
            prefix = "ai_interview_recordings/"
        
        response = s3_client.list_objects_v2(
            Bucket=AWS_S3_BUCKET,
            Prefix=prefix
        )
        
        objects = response.get('Contents', [])
        
        # Filter by candidate UUID if provided
        if candidate_uuid:
            objects = [obj for obj in objects if candidate_uuid in obj['Key']]
        
        return objects
        
    except Exception as e:
        logger.error(f"❌ Failed to list S3 objects: {e}")
        return []
