"""
MongoDB Database Connection and CRUD Operations
Local MongoDB setup for interview configuration management
"""
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from datetime import datetime
from typing import Optional, Dict, List
import os

# MongoDB Connection String (local for now)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "interview_bot")

# Global client instance
_client: Optional[MongoClient] = None
_db = None


def get_database():
    """Get MongoDB database instance"""
    global _client, _db
    
    if _db is None:
        try:
            _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            # Test connection
            _client.admin.command('ping')
            _db = _client[DB_NAME]
            print(f"✅ Connected to MongoDB: {DB_NAME}")
        except ConnectionFailure as e:
            print(f"❌ MongoDB Connection Failed: {e}")
            raise
    
    return _db


def init_database():
    """Initialize database with collections and indexes"""
    db = get_database()
    
    # Create indexes
    db.interview_configs.create_index("token", unique=True)
    db.interview_configs.create_index("isActive")
    db.available_voices.create_index("name", unique=True)
    db.interview_sessions.create_index("sessionId", unique=True)
    db.interview_sessions.create_index("configToken")
    
    # Seed available voices if empty
    if db.available_voices.count_documents({}) == 0:
        seed_voices(db)
    
    # Seed AI system settings if empty
    if db.ai_settings.count_documents({}) == 0:
        seed_ai_settings(db)
    
    # Seed default config if empty
    if db.interview_configs.count_documents({}) == 0:
        seed_default_config(db)
    
    print("✅ Database initialized with indexes and seed data")
    return db


def seed_voices(db):
    """Seed all 30 Gemini voices"""
    voices = [
        {"name": "Aoede", "style": "Breezy", "language": "en-US"},
        {"name": "Zephyr", "style": "Bright", "language": "en-US"},
        {"name": "Puck", "style": "Upbeat", "language": "en-US"},
        {"name": "Charon", "style": "Informative", "language": "en-US"},
        {"name": "Kore", "style": "Firm", "language": "en-US"},
        {"name": "Fenrir", "style": "Excitable", "language": "en-US"},
        {"name": "Leda", "style": "Youthful", "language": "en-US"},
        {"name": "Orus", "style": "Firm", "language": "en-US"},
        {"name": "Callirrhoe", "style": "Easy-going", "language": "en-US"},
        {"name": "Autonoe", "style": "Bright", "language": "en-US"},
        {"name": "Enceladus", "style": "Breathy", "language": "en-US"},
        {"name": "Iapetus", "style": "Clear", "language": "en-US"},
        {"name": "Umbriel", "style": "Easy-going", "language": "en-US"},
        {"name": "Algieba", "style": "Smooth", "language": "en-US"},
        {"name": "Despina", "style": "Smooth", "language": "en-US"},
    ]
    db.available_voices.insert_many(voices)
    print(f"✅ Seeded {len(voices)} voices")


def seed_ai_settings(db):
    """Seed default AI system settings (managed via DB only, not from admin UI)"""
    default_settings = {
        "settingsId": "default",
        
        # Voice/TTS tuning (DB-only)
        "voiceSpeed": 1.0,
        "voicePitch": 0,
        "temperature": 0.7,
        
        # Silence detection (DB-only)
        "silenceWarning1Seconds": 35,
        "silenceWarning2Seconds": 50,
        "silenceEndSeconds": 60,
        "autoEndDelaySeconds": 8,
        
        # Turn detection & latency (DB-only)
        "userSpeechEndDelayMs": 500,
        "aiResponseDelayMs": 200,
        "interruptionThresholdMs": 300,
        
        # AI behavior
        "maxQuestions": 10,
        
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }
    db.ai_settings.insert_one(default_settings)
    print("✅ Seeded default AI settings (DB-managed)")


def seed_default_config(db):
    """Seed a default configuration"""
    default_config = {
        "token": "default",
        
        # Voice selection (admin panel)
        "voiceName": "Aoede",
        "voiceStyle": "Professional and friendly",
        
        # Interview Info (admin panel)
        "companyName": "Demo Company",
        "jobRole": "Software Engineer",
        "candidateName": None,
        "candidateEmail": None,
        
        # New fields (admin panel)
        "language": "indian-english",
        "country": "India",
        "industryType": "Information Technology",
        "yearsOfExperience": "1-3",
        
        # Interview duration (admin panel)
        "durationMinutes": 30,
        
        # Custom prompt (admin panel, optional)
        "systemPrompt": None,
        
        # Proctoring (admin panel)
        "proctoring": {
            "enabled": True,
            "detectMultiplePeople": True,
            "detectPhone": True,
            "detectTabSwitch": True,
            "detectLookingAway": True,
            "strictness": "normal"
        },
        
        # Frontend UI (admin panel)
        "ui": {
            "appTitle": "AI Voice Interview",
            "logoUrl": None,
            "primaryColor": "#00ffd5",
            "backgroundColor": "#0a0a0a",
            "backgroundStyle": "grid",
            "welcomeMessage": "Click 'Start Interview' when you're ready to begin.",
            "showTimer": True,
            "darkModeDefault": True
        },
        
        # Recording (admin panel)
        "recording": {
            "audioEnabled": True,
            "screenEnabled": True,
            "autoDownload": False,
            "uploadToServer": True
        },
        
        # Status tracking
        "status": "active",  # active, scheduled, expired, completed
        "usageCount": 0,
        
        # Metadata
        "aiSettingsId": "default",  # Links to ai_settings collection
        "createdAt": datetime.utcnow(),
        "expiresAt": None,
        "isActive": True,
        "createdBy": "system"
    }
    
    db.interview_configs.insert_one(default_config)
    print("✅ Seeded default configuration (token: 'default')")


# ============================================================
# AI Settings (DB-managed only)
# ============================================================

def get_ai_settings(settings_id: str = "default") -> Optional[Dict]:
    """Get AI system settings"""
    db = get_database()
    settings = db.ai_settings.find_one(
        {"settingsId": settings_id},
        {"_id": 0}
    )
    return settings


def update_ai_settings(settings_id: str, update_data: Dict) -> bool:
    """Update AI system settings (via DB/Compass only)"""
    db = get_database()
    update_data["updatedAt"] = datetime.utcnow()
    result = db.ai_settings.update_one(
        {"settingsId": settings_id},
        {"$set": update_data}
    )
    return result.modified_count > 0


# ============================================================
# CRUD Operations for Interview Configs
# ============================================================

def get_config_by_token(token: str) -> Optional[Dict]:
    """Get interview configuration by token, merged with AI settings"""
    db = get_database()
    config = db.interview_configs.find_one(
        {"token": token, "isActive": True},
        {"_id": 0}
    )
    if config:
        # Merge AI settings
        ai_settings = get_ai_settings(config.get("aiSettingsId", "default"))
        if ai_settings:
            config["aiSettings"] = ai_settings
        
        # Increment usage count
        db.interview_configs.update_one(
            {"token": token},
            {"$inc": {"usageCount": 1}}
        )
    return config


def create_config(config_data: Dict) -> Dict:
    """Create new interview configuration"""
    db = get_database()
    config_data["createdAt"] = datetime.utcnow()
    config_data["isActive"] = True
    config_data["status"] = "active"
    config_data["usageCount"] = 0
    config_data["aiSettingsId"] = config_data.get("aiSettingsId", "default")
    
    result = db.interview_configs.insert_one(config_data)
    config_data["_id"] = str(result.inserted_id)
    return config_data


def update_config(token: str, update_data: Dict) -> bool:
    """Update existing configuration"""
    db = get_database()
    update_data["updatedAt"] = datetime.utcnow()
    
    result = db.interview_configs.update_one(
        {"token": token},
        {"$set": update_data}
    )
    return result.modified_count > 0


def delete_config(token: str) -> bool:
    """Soft delete configuration (set isActive=False)"""
    db = get_database()
    result = db.interview_configs.update_one(
        {"token": token},
        {"$set": {"isActive": False, "deletedAt": datetime.utcnow()}}
    )
    return result.modified_count > 0


def duplicate_config(token: str, new_token: str) -> Optional[Dict]:
    """Duplicate an existing configuration with a new token"""
    db = get_database()
    original = db.interview_configs.find_one({"token": token}, {"_id": 0})
    if not original:
        return None
    
    original["token"] = new_token
    original["createdAt"] = datetime.utcnow()
    original["usageCount"] = 0
    original["status"] = "active"
    original["isActive"] = True
    
    db.interview_configs.insert_one(original)
    return original


def list_configs(limit: int = 50, skip: int = 0) -> List[Dict]:
    """List all active configurations"""
    db = get_database()
    configs = list(db.interview_configs.find(
        {"isActive": True},
        {"_id": 0}
    ).sort("createdAt", -1).skip(skip).limit(limit))
    return configs


def get_all_voices() -> List[Dict]:
    """Get all available voices"""
    db = get_database()
    voices = list(db.available_voices.find({}, {"_id": 0}))
    return voices


# ============================================================
# Session & Latency Tracking
# ============================================================

def create_session(config_token: str, session_id: str) -> Dict:
    """Create new interview session for latency tracking"""
    db = get_database()
    session = {
        "configToken": config_token,
        "sessionId": session_id,
        "startedAt": datetime.utcnow(),
        "endedAt": None,
        "latencyLogs": [],
        "averageLatencyMs": 0,
        "maxLatencyMs": 0,
        "minLatencyMs": 0,
        "networkLogs": [],
        "totalQuestions": 0,
        "silenceWarnings": 0,
        "proctoringFlags": []
    }
    db.interview_sessions.insert_one(session)
    return session


def log_latency(session_id: str, user_end_time: int, ai_start_time: int) -> int:
    """Log latency measurement (times in milliseconds)
    Returns the calculated latency in ms
    """
    db = get_database()
    latency_ms = ai_start_time - user_end_time
    
    log_entry = {
        "timestamp": datetime.utcnow(),
        "userEndTime": user_end_time,
        "aiStartTime": ai_start_time,
        "latencyMs": latency_ms
    }
    
    # Add to logs and update stats
    db.interview_sessions.update_one(
        {"sessionId": session_id},
        {
            "$push": {"latencyLogs": log_entry},
        }
    )
    
    # Update min/max/avg
    session = db.interview_sessions.find_one({"sessionId": session_id})
    if session:
        logs = session.get("latencyLogs", [])
        latencies = [l["latencyMs"] for l in logs]
        if latencies:
            db.interview_sessions.update_one(
                {"sessionId": session_id},
                {
                    "$set": {
                        "averageLatencyMs": int(sum(latencies) / len(latencies)),
                        "maxLatencyMs": max(latencies),
                        "minLatencyMs": min(latencies)
                    }
                }
            )
    
    return latency_ms


def log_network_quality(session_id: str, speed_mbps: float, quality: str):
    """Log network quality measurement"""
    db = get_database()
    log_entry = {
        "timestamp": datetime.utcnow(),
        "speedMbps": speed_mbps,
        "quality": quality
    }
    db.interview_sessions.update_one(
        {"sessionId": session_id},
        {"$push": {"networkLogs": log_entry}}
    )


def end_session(session_id: str):
    """Mark session as ended"""
    db = get_database()
    db.interview_sessions.update_one(
        {"sessionId": session_id},
        {"$set": {"endedAt": datetime.utcnow()}}
    )


def get_session(session_id: str) -> Optional[Dict]:
    """Get session details"""
    db = get_database()
    return db.interview_sessions.find_one({"sessionId": session_id}, {"_id": 0})


def save_recording_url(session_id: str, candidate_uuid: str, file_type: str, s3_url: str, local_path: str):
    """
    Save recording URL to session document
    
    Args:
        session_id: Interview session UUID
        candidate_uuid: Candidate UUID
        file_type: Type of recording ('audio', 'combined_audio', 'screen', 'transcript')
        s3_url: S3 URL of the uploaded file
        local_path: Local file path (backup)
    """
    db = get_database()
    
    # Initialize recordings structure if not exists
    db.interview_sessions.update_one(
        {"sessionId": session_id},
        {
            "$set": {
                f"recordings.{file_type}Url": s3_url,
                f"localPaths.{file_type}": local_path,
                "candidateUuid": candidate_uuid,
                "updatedAt": datetime.utcnow()
            }
        },
        upsert=True
    )

