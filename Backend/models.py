"""
Pydantic Models for Interview Configuration System
Separated into: Admin Panel fields (ConfigCreate) vs DB-only fields (AISystemSettings)
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ============================================================
# DB-Only Settings (managed via MongoDB Compass, not admin UI)
# ============================================================

class AISystemSettings(BaseModel):
    """Technical AI settings - managed directly in DB, not from admin panel"""
    settingsId: str = Field(default="default")
    voiceSpeed: float = Field(default=1.0, ge=0.25, le=4.0)
    voicePitch: int = Field(default=0, ge=-20, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    silenceWarning1Seconds: int = Field(default=35, ge=10, le=120)
    silenceWarning2Seconds: int = Field(default=50, ge=15, le=150)
    silenceEndSeconds: int = Field(default=60, ge=20, le=180)
    autoEndDelaySeconds: int = Field(default=8, ge=3, le=30)
    userSpeechEndDelayMs: int = Field(default=500, ge=100, le=2000)
    aiResponseDelayMs: int = Field(default=200, ge=0, le=1000)
    interruptionThresholdMs: int = Field(default=300, ge=100, le=1000)
    maxQuestions: int = Field(default=10, ge=1, le=50)


# ============================================================
# Admin Panel Models (shown in admin UI)
# ============================================================

class ProctoringSettings(BaseModel):
    """Proctoring configuration"""
    enabled: bool = True
    detectMultiplePeople: bool = True
    detectPhone: bool = True
    detectTabSwitch: bool = True
    detectLookingAway: bool = True
    strictness: str = Field(default="normal", pattern="^(lenient|normal|strict)$")


class UISettings(BaseModel):
    """Frontend UI settings"""
    appTitle: str = Field(default="AI Voice Interview")
    logoUrl: Optional[str] = None
    primaryColor: str = Field(default="#00ffd5")
    backgroundColor: str = Field(default="#0a0a0a")
    backgroundStyle: str = Field(default="grid", pattern="^(grid|gradient|solid)$")
    welcomeMessage: str = Field(default="Click 'Start Interview' when you're ready.")
    showTimer: bool = True
    darkModeDefault: bool = True


class RecordingSettings(BaseModel):
    """Recording configuration"""
    audioEnabled: bool = True
    screenEnabled: bool = True
    autoDownload: bool = False
    uploadToServer: bool = True


class ConfigCreate(BaseModel):
    """Model for creating new config from admin panel
    Only business-facing fields - technical AI settings come from DB
    """
    token: str
    
    # Voice (just name + style, speed/pitch from DB)
    voiceName: str = Field(default="Aoede")
    voiceStyle: Optional[str] = Field(default=None)
    
    # Interview Info
    companyName: Optional[str] = None
    jobRole: str = Field(default="Software Engineer")
    candidateName: Optional[str] = None
    candidateEmail: Optional[str] = None
    
    # New business fields
    language: str = Field(default="indian-english", description="Interview language")
    country: str = Field(default="India")
    industryType: str = Field(default="Information Technology")
    yearsOfExperience: str = Field(default="1-3")
    
    # Duration
    durationMinutes: int = Field(default=30, ge=1, le=180)
    
    # Custom prompt (optional override)
    systemPrompt: Optional[str] = None
    
    # Proctoring
    proctoring: Optional[ProctoringSettings] = None
    
    # UI
    ui: Optional[UISettings] = None
    
    # Recording
    recording: Optional[RecordingSettings] = None


class ConfigUpdate(BaseModel):
    """Model for updating config from admin panel"""
    voiceName: Optional[str] = None
    voiceStyle: Optional[str] = None
    companyName: Optional[str] = None
    jobRole: Optional[str] = None
    candidateName: Optional[str] = None
    candidateEmail: Optional[str] = None
    language: Optional[str] = None
    country: Optional[str] = None
    industryType: Optional[str] = None
    yearsOfExperience: Optional[str] = None
    durationMinutes: Optional[int] = None
    systemPrompt: Optional[str] = None
    proctoring: Optional[ProctoringSettings] = None
    ui: Optional[UISettings] = None
    recording: Optional[RecordingSettings] = None
    isActive: Optional[bool] = None


class LatencyLog(BaseModel):
    """Latency log entry"""
    sessionId: str
    userEndTime: int = Field(..., description="Timestamp when user stopped speaking (ms)")
    aiStartTime: int = Field(..., description="Timestamp when AI started responding (ms)")


class VoiceInfo(BaseModel):
    """Voice information"""
    name: str
    style: str
    language: str = "en-US"
