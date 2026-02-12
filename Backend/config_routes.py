"""
API Routes for Interview Configuration Management
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import uuid

from database import (
    get_config_by_token, create_config, update_config, delete_config,
    list_configs, get_all_voices, init_database, get_ai_settings,
    create_session, log_latency, get_session, end_session, log_network_quality,
    duplicate_config
)
from models import (
    ConfigCreate, ConfigUpdate, 
    LatencyLog, VoiceInfo
)

router = APIRouter(prefix="/api", tags=["Configuration"])


# ============================================================
# Configuration Endpoints
# ============================================================

@router.get("/config/{token}")
async def get_config(token: str):
    """
    Get interview configuration by token.
    Returns config merged with AI settings from separate collection.
    """
    config = get_config_by_token(token)
    if not config:
        raise HTTPException(status_code=404, detail=f"Config not found for token: {token}")
    return config


@router.get("/configs")
async def list_all_configs(
    limit: int = Query(default=50, le=100),
    skip: int = Query(default=0, ge=0)
):
    """List all active configurations (for admin panel)"""
    configs = list_configs(limit=limit, skip=skip)
    return {"configs": configs, "count": len(configs)}


@router.post("/config")
async def create_new_config(config_data: ConfigCreate):
    """Create a new interview configuration"""
    # Check if token already exists
    existing = get_config_by_token(config_data.token)
    if existing:
        raise HTTPException(status_code=400, detail=f"Token '{config_data.token}' already exists")
    
    # Convert to dict and create
    config_dict = config_data.model_dump(exclude_none=True)
    
    # Set defaults for missing nested objects
    defaults = {
        "proctoring": {"enabled": True, "detectMultiplePeople": True, "detectPhone": True, "detectTabSwitch": True, "detectLookingAway": True, "strictness": "normal"},
        "ui": {"appTitle": "AI Voice Interview", "logoUrl": None, "primaryColor": "#00ffd5", "backgroundColor": "#0a0a0a", "backgroundStyle": "grid", "welcomeMessage": "Click 'Start Interview' when you're ready.", "showTimer": True, "darkModeDefault": True},
        "recording": {"audioEnabled": True, "screenEnabled": True, "autoDownload": True, "uploadToServer": True}
    }
    
    for key, default_value in defaults.items():
        if key not in config_dict or config_dict[key] is None:
            config_dict[key] = default_value
    
    result = create_config(config_dict)
    return {"status": "created", "token": config_data.token}


@router.put("/config/{token}")
async def update_existing_config(token: str, update_data: ConfigUpdate):
    """Update an existing configuration"""
    existing = get_config_by_token(token)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Config not found for token: {token}")
    
    # Only update fields that are provided
    update_dict = update_data.model_dump(exclude_none=True)
    
    # Merge nested objects instead of replacing
    for key in ["proctoring", "ui", "recording"]:
        if key in update_dict and key in existing:
            existing[key].update(update_dict[key])
            update_dict[key] = existing[key]
    
    success = update_config(token, update_dict)
    if success:
        return {"status": "updated", "token": token}
    else:
        raise HTTPException(status_code=500, detail="Update failed")


@router.delete("/config/{token}")
async def deactivate_config(token: str):
    """Deactivate a configuration (soft delete)"""
    success = delete_config(token)
    if success:
        return {"status": "deactivated", "token": token}
    else:
        raise HTTPException(status_code=404, detail=f"Config not found for token: {token}")


@router.post("/config/{token}/duplicate")
async def duplicate_existing_config(token: str):
    """Duplicate an existing configuration with a new UUID token"""
    new_token = str(uuid.uuid4())
    result = duplicate_config(token, new_token)
    if result:
        return {"status": "duplicated", "originalToken": token, "newToken": new_token}
    else:
        raise HTTPException(status_code=404, detail=f"Config not found for token: {token}")


# ============================================================
# Voice Endpoints
# ============================================================

@router.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    """Get all available Gemini voices"""
    voices = get_all_voices()
    return voices


# ============================================================
# AI Settings (DB-only, read endpoint for internal use)
# ============================================================

@router.get("/ai-settings")
async def get_system_settings(settings_id: str = "default"):
    """Get AI system settings (managed via DB/Compass only)"""
    settings = get_ai_settings(settings_id)
    if not settings:
        raise HTTPException(status_code=404, detail="AI settings not found")
    return settings


# ============================================================
# Session & Latency Tracking Endpoints
# ============================================================

@router.post("/session/start")
async def start_session(config_token: str):
    """Start a new interview session for latency tracking"""
    session_id = str(uuid.uuid4())
    session = create_session(config_token, session_id)
    return {"sessionId": session_id, "configToken": config_token}


@router.post("/latency/log")
async def log_latency_measurement(latency_data: LatencyLog):
    """
    Log a latency measurement.
    Times should be in milliseconds (integer).
    Returns the calculated latency.
    """
    latency_ms = log_latency(
        session_id=latency_data.sessionId,
        user_end_time=latency_data.userEndTime,
        ai_start_time=latency_data.aiStartTime
    )
    return {"latencyMs": latency_ms, "sessionId": latency_data.sessionId}


@router.post("/network/log")
async def log_network(session_id: str, speed_mbps: float, quality: str):
    """Log network quality measurement"""
    log_network_quality(session_id, speed_mbps, quality)
    return {"status": "logged"}


@router.get("/session/{session_id}")
async def get_session_details(session_id: str):
    """Get session details including latency stats"""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/session/{session_id}/end")
async def end_interview_session(session_id: str):
    """Mark session as ended"""
    end_session(session_id)
    return {"status": "ended", "sessionId": session_id}


# ============================================================
# Database Initialization
# ============================================================

@router.post("/init-db")
async def initialize_database():
    """Initialize database with collections and seed data"""
    try:
        init_database()
        return {"status": "initialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
