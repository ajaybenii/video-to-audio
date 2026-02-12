from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import asyncio
import json
import os
import logging
from websockets.legacy.client import connect
from datetime import datetime
import time
from collections import deque
import uuid
from pathlib import Path
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# Import config routes and database
from config_routes import router as config_router
from database import get_config_by_token, init_database, create_session, log_latency

load_dotenv(override=True)

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("interview-bot")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Import for service account authentication
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Configuration (from environment variables)
PROJECT_ID = os.getenv("PROJECT_ID", "sqy-prod")
LOCATION = os.getenv("LOCATION", "us-central1")
MODEL_ID = "gemini-live-2.5-flash-native-audio"
MODEL = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"
HOST = f"wss://{LOCATION}-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"

# 🔥 PRODUCTION SETTINGS
MAX_CONCURRENT_CONNECTIONS = int(os.getenv("MAX_CONCURRENT_CONNECTIONS", "1000"))
CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", "1800"))  # 30 minutes

INTERVIEW_PROMPT = """
You are conducting a real-time technical interview for a Software Engineer position.
You are based in INDIA and conducting this interview in Indian Standard Time (IST, UTC+5:30).

You can hear and also see the candidate through audio and video.

# 🔴 PROCTORING & MONITORING (HIGH PRIORITY):
You MUST continuously monitor the video feed and IMMEDIATELY warn the candidate if you detect:

1. **Multiple People Detected**: If you see more than ONE person in the frame:
   - Immediately say: "I notice there might be someone else in the room. For interview integrity, please ensure you are alone. This will be noted."

2. **Mobile Phone Usage**: If you see the candidate using or looking at a mobile phone:
   - Immediately say: "I noticed you looking at your phone. Please keep your phone away during the interview. Using external devices is not allowed."

3. **Candidate Not Visible**: If the candidate is not visible or has moved out of frame:
   - Immediately say: "I can't see you on the screen. Please adjust your camera so I can see you clearly."

4. **Looking Away / Not Focused**: If the candidate is frequently looking away from the screen (looking left, right, up, or down repeatedly):
   - Say: "I notice you're looking away from the screen. Please focus on the interview and maintain eye contact with the camera."

5. **Suspicious Behavior**: If you see any suspicious behavior like reading from another screen, someone whispering, or unusual movements:
   - Say: "I noticed some unusual activity. Please remember this is a proctored interview and any unfair means will be recorded."

6. **Tab Switching / Distraction**: If the candidate appears distracted or seems to be reading something off-screen:
   - Say: "It seems like you might be looking at something else. Please give your full attention to the interview."

# 🌐 NETWORK MONITORING:
If you receive a message indicating the candidate's network quality is POOR:
- Say: "I'm noticing some connectivity issues on your end. If possible, please move to a location with better internet connection for a smoother interview experience."

# ⏱️ SILENT USER DETECTION:
The system will send you messages about candidate silence. Respond appropriately:
- If you receive "[SYSTEM] I am waiting for your response": 
  Say: "I am waiting for your response."
- If you receive "[SYSTEM] Candidate silent for {SILENCE_WARNING_2} seconds - FINAL WARNING - Interview will end in {AUTO_END_DELAY} seconds":
  Say firmly: "If you do not respond, we will end the interview shortly."
- If you receive "[SYSTEM] Ending interview due to no response":
  Say: "Since there has been no response, we are ending this interview session now. Thank you for your time."
- If you receive "[SYSTEM] Interview time limit ({DURATION} minutes) reached. Ending interview.":
  Say: "We have reached the time limit for this interview. Thank you for your time today. The interview is now complete."

# 🔄 IMPORTANT - CONTINUING INTERVIEW:
If the candidate speaks or responds after any warning (including the final warning), you MUST:
- IMMEDIATELY continue the interview as normal
- Do NOT say "the interview has ended" or "we've reached the time limit" (unless the 30-minute timer actually reached zero)
- Do NOT refuse to continue - just pick up where you left off
- Simply acknowledge their response and continue with the next question
The silence warnings are just prompts - if the user responds, the interview continues!

# ⚠️ IMPORTANT: Issue warnings in a FIRM but PROFESSIONAL tone. Do not be rude, but be clear that violations are being noted.

# Interview Structure:
1. Greet the candidate appropriately based on Indian time:
   - Morning (6 AM - 12 PM IST): "Good morning"
   - Afternoon (12 PM - 5 PM IST): "Good afternoon"  
   - Evening (5 PM - 9 PM IST): "Good evening"
   - Night (9 PM - 6 AM IST): "Hello"

2. Ask candidate to introduce themselves
3. Ask 3 technical questions about:
   - Data structures and algorithms
   - System design
   - Problem-solving approach
4. Ask 2 behavioral questions
5. Close the interview professionally

# Visual Observation Rules:
- You can see the candidate through video
- Answer visual questions ONLY based on what is clearly visible
- If something is not clearly visible, say you are not certain
- Do not guess or assume

# Communication Rules:
- Be professional but friendly (Indian professional context)
- Listen carefully and ask follow-up questions
- Keep responses concise
- Encourage the candidate when they do well
- Use natural, conversational language
- Speak clearly in English (Indian candidates may have regional accents - be patient)
"""

# Service Account Authentication
def get_access_token():
    """Get access token using service account credentials"""
    try:
        # Load credentials from environment or file
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if credentials_path and os.path.exists(credentials_path):
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            # Load from JSON string in environment variable
            credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
            if credentials_json:
                credentials_info = json.loads(credentials_json)
                credentials = service_account.Credentials.from_service_account_info(
                    credentials_info,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            else:
                logger.error("No credentials found in environment")
                return None
        
        # Refresh token
        credentials.refresh(Request())
        return credentials.token
    except Exception as e:
        logger.error(f"Error getting access token: {e}")
        return None

# Lifespan context manager (replaces deprecated @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=" * 60)
    logger.info("VOICE + VIDEO INTERVIEW BOT API")
    logger.info("=" * 60)
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Max Connections: {MAX_CONCURRENT_CONNECTIONS}")
    logger.info(f"Connection Timeout: {CONNECTION_TIMEOUT}s")
    logger.info(f"Log Level: {LOG_LEVEL}")
    logger.info(f"Video Support: ENABLED")
    logger.info("=" * 60)
    logger.info("Server Ready!")
    yield
    # Shutdown
    logger.info("Server shutting down...")

# Initialize FastAPI
app = FastAPI(
    title="Voice + Video Interview Bot API - Production",
    description="High-performance interview bot with unlimited rate limits",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include config routes
app.include_router(config_router)

# Connection Management
class ConnectionManager:
    def __init__(self):
        self.active_connections = 0
        self.total_connections = 0
        self.connection_history = deque(maxlen=100)
        self.token_cache = None
        self.token_expiry = None
        self.start_time = datetime.now()
    
    def can_accept_connection(self) -> bool:
        return self.active_connections < MAX_CONCURRENT_CONNECTIONS
    
    def add_connection(self):
        self.active_connections += 1
        self.total_connections += 1
        self.connection_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'connected',
            'active': self.active_connections,
            'total': self.total_connections
        })
    
    def remove_connection(self):
        self.active_connections = max(0, self.active_connections - 1)
        self.connection_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'disconnected',
            'active': self.active_connections
        })
    
    def get_cached_token(self):
        """Cache token to optimize performance"""
        now = datetime.now()
        if self.token_cache and self.token_expiry and now < self.token_expiry:
            return self.token_cache
        
        # Get new token
        token = get_access_token()
        if token:
            self.token_cache = token
            # Cache for 50 minutes (tokens valid for 1 hour)
            from datetime import timedelta
            self.token_expiry = now + timedelta(minutes=50)
        return token
    
    def get_stats(self):
        uptime = datetime.now() - self.start_time
        return {
            'active_connections': self.active_connections,
            'total_connections': self.total_connections,
            'max_capacity': MAX_CONCURRENT_CONNECTIONS,
            'available_slots': MAX_CONCURRENT_CONNECTIONS - self.active_connections,
            'uptime_seconds': int(uptime.total_seconds()),
            'uptime_formatted': str(uptime).split('.')[0]
        }

manager = ConnectionManager()

async def relay_messages(ws_client: WebSocket, ws_google):
    """Handle bidirectional message relay between client and Gemini"""
    
    # Store session resumption handle
    session_handle = None
    
    async def client2server(source: WebSocket, target):
        """Browser → Gemini (audio + video)"""
        msg_count = 0
        audio_chunk_count = 0
        try:
            while True:
                message = await source.receive_text()
                msg_count += 1
                data = json.loads(message)
                
                # Logging (only in debug mode)
                if 'realtimeInput' in data:
                    audio_chunk_count += 1
                    if audio_chunk_count % 100 == 0:
                        logger.debug(f"Media chunks sent: {audio_chunk_count}")
                else:
                    logger.debug(f"Browser→Gemini message #{msg_count}")
                
                await target.send(message)
        except WebSocketDisconnect:
            logger.debug("Client disconnected from relay")
        except Exception as e:
            logger.error(f"Error client2server: {e}")
    
    async def server2client(source, target: WebSocket):
        """Gemini → Browser"""
        nonlocal session_handle
        msg_count = 0
        try:
            async for message in source:
                msg_count += 1
                data = json.loads(message.decode('utf-8'))
                
                # Handle session resumption updates
                if 'sessionResumptionUpdate' in data:
                    update = data['sessionResumptionUpdate']
                    if update.get('resumable') and update.get('newHandle'):
                        session_handle = update['newHandle']
                        logger.debug("Session resumption handle updated")
                
                # Handle GoAway message (connection will terminate soon)
                if 'goAway' in data:
                    time_left = data['goAway'].get('timeLeft', 'unknown')
                    logger.warning(f"Connection will close in {time_left}. Resumption handle available: {bool(session_handle)}")
                
                # Detailed logging in debug mode
                if 'serverContent' in data:
                    content = data['serverContent']
                    
                    if 'modelTurn' in content:
                        logger.debug("AI Speaking")
                    
                    if 'outputTranscription' in content:
                        text = content['outputTranscription'].get('text', '')
                        logger.debug(f"AI said: {text}")
                    
                    if 'inputTranscription' in content:
                        text = content['inputTranscription'].get('text', '')
                        is_final = content['inputTranscription'].get('isFinal', False)
                        if is_final:
                            logger.debug(f"User said: {text}")
                    
                    if 'generationComplete' in content:
                        logger.debug("Generation complete")
                
                elif 'setupComplete' in data:
                    logger.debug("Setup complete")
                
                await target.send_text(message.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error server2client: {e}")
    
    # Set timeout for the entire connection
    try:
        await asyncio.wait_for(
            asyncio.gather(
                client2server(ws_client, ws_google),
                server2client(ws_google, ws_client),
                return_exceptions=True
            ),
            timeout=CONNECTION_TIMEOUT
        )
    except asyncio.TimeoutError:
        print(f"⏰ Connection timeout after {CONNECTION_TIMEOUT} seconds")

@app.get("/")
async def root():
    """API information endpoint"""
    stats = manager.get_stats()
    return {
        "status": "online",
        "service": "Voice + Video Interview Bot API",
        "version": "2.0.0",
        "model": MODEL_ID,
        "features": ["audio", "video", "transcription", "unlimited-rate-limits"],
        "websocket_endpoint": "/ws/interview",
        **stats
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring and load balancers"""
    stats = manager.get_stats()
    is_healthy = stats['active_connections'] < MAX_CONCURRENT_CONNECTIONS
    
    return {
        "status": "healthy" if is_healthy else "at_capacity",
        "video_support": True,
        "rate_limits": "unlimited",
        **stats
    }

@app.get("/stats")
async def get_stats():
    """Detailed statistics endpoint"""
    stats = manager.get_stats()
    return {
        **stats,
        "recent_activity": list(manager.connection_history)[-20:],
        "configuration": {
            "max_concurrent_connections": MAX_CONCURRENT_CONNECTIONS,
            "connection_timeout": CONNECTION_TIMEOUT,
            "model": MODEL_ID,
            "location": LOCATION
        }
    }

@app.websocket("/ws/interview")
async def websocket_interview(websocket: WebSocket, token: str = Query(default="default")):
    """Main WebSocket endpoint for voice + video interview with dynamic config"""
    
    # Check capacity
    if not manager.can_accept_connection():
        await websocket.close(code=1008, reason="Server at capacity")
        logger.warning(f"Connection rejected - At capacity ({manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS})")
        return
    
    await websocket.accept()
    manager.add_connection()
    
    connection_id = manager.total_connections
    session_id = str(uuid.uuid4())
    
    logger.info(f"Client #{connection_id} connected ({manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS} active) - Token: {token}")
    
    # Load config from MongoDB
    config = get_config_by_token(token)
    if not config:
        logger.warning(f"Config not found for token: {token}, using defaults")
        config = get_config_by_token("default")
    
    if not config:
        logger.error("No config found and no default config available")
        manager.remove_connection()
        await websocket.close(code=1011, reason="Configuration not found")
        return
    
    # Extract settings from config
    # AI settings come from separate collection, merged by get_config_by_token()
    ai_settings = config.get("aiSettings", {})
    proctoring = config.get("proctoring", {})
    
    # Get voice name (flat field now, not nested)
    voice_name = config.get("voiceName", "Aoede")
    voice_style = config.get("voiceStyle", "")
    
    # AI tuning from ai_settings (DB-managed)
    voice_speed = ai_settings.get("voiceSpeed", 1.0)
    voice_pitch = ai_settings.get("voicePitch", 0)
    temperature = ai_settings.get("temperature", 0.7)
    
    # Timing from ai_settings (DB-managed)
    duration_minutes = config.get("durationMinutes", 30)
    silence_warning1 = ai_settings.get("silenceWarning1Seconds", 35)
    silence_warning2 = ai_settings.get("silenceWarning2Seconds", 50)
    silence_end = ai_settings.get("silenceEndSeconds", 60)
    
    # Turn detection from ai_settings (DB-managed, in ms)
    user_speech_end_delay = ai_settings.get("userSpeechEndDelayMs", 500)
    ai_response_delay = ai_settings.get("aiResponseDelayMs", 200)
    
    # Language setting
    language = config.get("language", "indian-english")
    
    # Build dynamic system prompt
    custom_prompt = config.get("systemPrompt")
    if custom_prompt:
        system_prompt = custom_prompt
    else:
        system_prompt = generate_dynamic_prompt(
            config=config,
            ai_settings=ai_settings,
            proctoring=proctoring
        )
    
    # Build timing/turn detection for frontend
    timing_for_frontend = {
        "durationMinutes": duration_minutes,
        "silenceWarning1Seconds": silence_warning1,
        "silenceWarning2Seconds": silence_warning2,
        "silenceEndSeconds": silence_end,
        "autoEndDelaySeconds": ai_settings.get("autoEndDelaySeconds", 8)
    }
    turn_detection_for_frontend = {
        "userSpeechEndDelayMs": user_speech_end_delay,
        "aiResponseDelayMs": ai_response_delay,
        "interruptionThresholdMs": ai_settings.get("interruptionThresholdMs", 300)
    }
    
    # Generate a candidate UUID for tracking recordings in S3
    candidate_uuid = str(uuid.uuid4())
    
    # Send config to client for frontend settings
    config_message = {
        "type": "config",
        "config": {
            "timing": timing_for_frontend,
            "turnDetection": turn_detection_for_frontend,
            "ui": config.get("ui", {}),
            "recording": config.get("recording", {}),
            "interview": {
                "companyName": config.get("companyName"),
                "jobRole": config.get("jobRole"),
                "candidateName": config.get("candidateName"),
                "language": language
            },
            "sessionId": session_id,
            "candidateUuid": candidate_uuid
        }
    }
    await websocket.send_json(config_message)
    
    # Create session for latency tracking
    try:
        create_session(token, session_id)
    except Exception as e:
        logger.warning(f"Failed to create session for tracking: {e}")
    
    # Get cached token for better performance
    access_token = manager.get_cached_token()
    
    if not access_token:
        logger.error("Failed to get access token")
        manager.remove_connection()
        await websocket.close(code=1011, reason="Authentication failed")
        return
    
    try:
        async with connect(
            HOST,
            extra_headers={'Authorization': f'Bearer {access_token}'},
            ping_interval=20,
            ping_timeout=10,
            max_size=10_000_000  # 10MB max message size for video
        ) as ws_google:
            # Setup with dynamic audio and video support
            initial_request = {
                "setup": {
                    "model": MODEL,
                    "generationConfig": {
                        "temperature": temperature,
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {
                                    "voiceName": voice_name
                                }
                            }
                        }
                    },
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "input_audio_transcription": {},
                    "output_audio_transcription": {},
                    # Enable context window compression for unlimited session time
                    "context_window_compression": {
                        "sliding_window": {},
                        "trigger_tokens": 50000
                    },
                    # Enable session resumption for handling connection resets
                    "session_resumption": {}
                }
            }
            
            await ws_google.send(json.dumps(initial_request))
            
            logger.info(f"Client #{connection_id} - AI initialized with voice: {voice_name}, temp: {temperature}")
            
            await relay_messages(websocket, ws_google)
            
    except WebSocketDisconnect:
        logger.info(f"Client #{connection_id} disconnected")
    except Exception as e:
        logger.error(f"Client #{connection_id} error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
    finally:
        manager.remove_connection()
        logger.info(f"Client #{connection_id} session ended. Active: {manager.active_connections}/{MAX_CONCURRENT_CONNECTIONS}")


def generate_dynamic_prompt(config: dict, ai_settings: dict, proctoring: dict) -> str:
    """Generate system prompt dynamically based on config and language"""
    
    company_name = config.get("companyName", "the company")
    job_role = config.get("jobRole", "Software Engineer")
    candidate_name = config.get("candidateName", "")
    language = config.get("language", "indian-english")
    country = config.get("country", "India")
    industry = config.get("industryType", "Information Technology")
    experience = config.get("yearsOfExperience", "1-3")
    
    duration_minutes = config.get("durationMinutes", 30)
    silence_warning1 = ai_settings.get("silenceWarning1Seconds", 35)
    silence_warning2 = ai_settings.get("silenceWarning2Seconds", 50)
    silence_end = ai_settings.get("silenceEndSeconds", 60)
    max_questions = ai_settings.get("maxQuestions", 10)
    
    # Language instructions — MUST be at the very top of the prompt
    if language == "casual-hindi":
        lang_instruction = """# ⚠️ CRITICAL MANDATORY LANGUAGE RULE — YOU MUST FOLLOW THIS:
YOU MUST SPEAK ONLY IN CASUAL HINDI (Hinglish). DO NOT SPEAK IN ENGLISH.
This is the #1 most important rule. Every single sentence you say MUST be in Hindi/Hinglish.

## Hindi Language Rules:
- ALWAYS speak in casual, everyday Hindi (Hinglish) — NOT formal/shudh Hindi
- DO NOT use English for sentences. Only use English for technical terms like 'array', 'database', 'API', 'system design', etc.
- Your greeting MUST be in Hindi, e.g., "Namaste! Main aapka interview lunga aaj."
- Example questions: "Acha, toh mujhe batao ki tumne apne last project mein kya kya kiya?"
- Example follow-up: "Hmm interesting, toh usme kaunsa tech stack use kiya tha?"
- Be natural and friendly, jaise ek colleague se baat kar rahe ho chai pe
- Even if the candidate replies in English, YOU MUST continue speaking in Hindi/Hinglish
- NEVER switch to English. ALWAYS stay in Hindi/Hinglish no matter what.

REMEMBER: SPEAK IN HINDI. NOT ENGLISH. THIS IS NON-NEGOTIABLE."""
    else:  # indian-english (default)
        lang_instruction = """# 🗣️ LANGUAGE: INDIAN ENGLISH
- Speak in clear, professional Indian English
- Use natural Indian expressions and phrasing
- Be professional but warm and approachable"""
    
    # Experience-based question difficulty
    exp_instruction = ""
    if experience == "0-1":
        exp_instruction = "Ask beginner-friendly questions. Focus on fundamentals, basic concepts, and willingness to learn."
    elif experience == "1-3":
        exp_instruction = "Ask intermediate questions. Focus on practical experience, problem-solving, and coding skills."
    elif experience == "3-5":
        exp_instruction = "Ask mid-level questions. Include system design basics, architecture decisions, and leadership potential."
    elif experience == "5-10":
        exp_instruction = "Ask senior-level questions. Focus on system design, architecture, mentoring, and strategic thinking."
    elif experience == "10+":
        exp_instruction = "Ask principal/lead-level questions. Focus on large-scale system design, organizational impact, and technical vision."
    
    # Build proctoring rules
    proctoring_rules = ""
    if proctoring.get("enabled", True):
        rules = []
        if proctoring.get("detectMultiplePeople", True):
            rules.append('1. **Multiple People Detected**: If you see more than ONE person in the frame, warn in the interview language: "I notice there might be someone else in the room. Please ensure you are alone."')
        if proctoring.get("detectPhone", True):
            rules.append('2. **Mobile Phone Usage**: If you see a phone, warn in the interview language: "Please keep your phone away during the interview."')
        if proctoring.get("detectLookingAway", True):
            rules.append('3. **Looking Away**: If candidate frequently looks away, warn in the interview language: "Please focus on the interview and maintain eye contact."')
        if proctoring.get("detectTabSwitch", True):
            rules.append('4. **Tab Switching**: If distracted, warn in the interview language: "Please give your full attention to the interview."')
        proctoring_rules = "\n".join(rules)
    
    # Build prompt — LANGUAGE INSTRUCTION GOES FIRST
    prompt = f"""{lang_instruction}

You are conducting a real-time technical interview for {company_name} for the position of {job_role}.
Industry: {industry} | Country: {country} | Expected Experience: {experience} years

You can hear and also see the candidate through audio and video.
{"The candidate's name is " + candidate_name + "." if candidate_name else ""}

# 🎯 EXPERIENCE-BASED DIFFICULTY:
{exp_instruction}

# 🔴 PROCTORING & MONITORING:
{proctoring_rules if proctoring_rules else "Proctoring is disabled for this interview."}

# ⏱️ SILENT USER DETECTION:
The system will send you messages about candidate silence. Respond appropriately:
- If you receive "[SYSTEM] I am waiting for your response": 
  Say: "I am waiting for your response."
- If you receive "[SYSTEM] Candidate silent for {silence_warning2} seconds - FINAL WARNING":
  Say firmly: "If you do not respond, we will end the interview shortly."
- If you receive "[SYSTEM] Ending interview due to no response":
  Say: "Since there has been no response, we are ending this interview session now."
- If you receive "[SYSTEM] Interview time limit ({duration_minutes} minutes) reached":
  Say: "We have reached the {duration_minutes}-minute time limit. The interview is now complete."

# 🔄 IMPORTANT:
If the candidate speaks after any warning, IMMEDIATELY continue the interview normally.

# Interview Structure:
1. Greet the candidate appropriately
2. Ask candidate to introduce themselves
3. Ask up to {max_questions} questions appropriate for their experience level
4. Close the interview professionally

# Communication Rules:
- Be professional but friendly
- Listen carefully and ask follow-up questions
- Keep responses concise
- Encourage good answers
- Use natural, conversational language
"""
    return prompt

# Startup event moved to lifespan context manager above

# Network Info Endpoint for latency measurement
@app.get("/api/network-info")
async def network_info():
    """
    Returns server timestamp for client-side latency calculation.
    This endpoint is used by the frontend to measure network quality.
    """
    return {
        "timestamp": int(time.time() * 1000),  # milliseconds
        "status": "ok",
        "server_time": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    stats = manager.get_stats()
    return {
        "status": "healthy",
        "active_connections": stats['active_connections'],
        "uptime": stats['uptime_formatted']
    }

# ============================================================
# SPEED TEST API (Business Reusable)
# ============================================================

# Store speed test results for analytics
speed_test_results = []

@app.get("/api/speed-test/download")
async def speed_test_download(bytes: int = 100000):
    """
    Serve binary data for speed test.
    Client downloads this and measures time to calculate bandwidth.
    Args:
        bytes: Size of test data (default 100KB, max 1MB)
    """
    # Limit max size to 1MB to prevent abuse
    size = min(bytes, 1000000)
    # Generate random bytes for download
    data = os.urandom(size)
    
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(size),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "X-Speed-Test": "true"
        }
    )

@app.post("/api/speed-test/report")
async def speed_test_report(
    speed_mbps: float,
    quality: str = "unknown",
    user_agent: str = None
):
    """
    Report speed test result for analytics.
    Args:
        speed_mbps: Measured download speed in Mbps
        quality: Network quality (good/fair/poor)
        user_agent: Client user agent
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "speed_mbps": speed_mbps,
        "quality": quality,
        "user_agent": user_agent
    }
    speed_test_results.append(result)
    
    # Keep only last 1000 results in memory
    if len(speed_test_results) > 1000:
        speed_test_results.pop(0)
    
    logger.info(f"Speed test reported: {speed_mbps:.1f} Mbps ({quality})")
    return {"status": "recorded", "speed_mbps": speed_mbps}

@app.get("/api/speed-test/stats")
async def speed_test_stats():
    """
    Get speed test analytics (last 24 hours summary).
    """
    if not speed_test_results:
        return {"count": 0, "avg_speed": 0, "min_speed": 0, "max_speed": 0}
    
    speeds = [r["speed_mbps"] for r in speed_test_results]
    return {
        "count": len(speeds),
        "avg_speed": round(sum(speeds) / len(speeds), 2),
        "min_speed": round(min(speeds), 2),
        "max_speed": round(max(speeds), 2),
        "quality_distribution": {
            "good": sum(1 for r in speed_test_results if r["quality"] == "good"),
            "fair": sum(1 for r in speed_test_results if r["quality"] == "fair"),
            "poor": sum(1 for r in speed_test_results if r["quality"] == "poor")
        }
    }

# ============================================================
# RECORDING & ANALYSIS ENDPOINTS
# ============================================================

# Directory paths for recordings
RECORDINGS_DIR = Path(__file__).parent / "recordings"
AUDIO_DIR = RECORDINGS_DIR / "audio"
AUDIO_USER_DIR = AUDIO_DIR / "user"
AUDIO_COMBINED_DIR = AUDIO_DIR / "combined"
SCREEN_DIR = RECORDINGS_DIR / "screen"
TRANSCRIPTS_DIR = RECORDINGS_DIR / "transcripts"

# Ensure directories exist
for dir_path in [AUDIO_DIR, AUDIO_USER_DIR, AUDIO_COMBINED_DIR, SCREEN_DIR, TRANSCRIPTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configure Vertex AI (uses sqy-prod.json service account)
vertexai.init(project=PROJECT_ID, location=LOCATION)

@app.post("/api/upload-recording")
async def upload_recording(
    file: UploadFile = File(...),
    recording_type: str = "audio",  # audio, combined_audio, or screen
    session_id: str = None,  # Session UUID (if not provided, generate one)
    candidate_uuid: str = None  # Candidate UUID for S3 folder organization
):
    """
    Upload audio or screen recording DIRECTLY to S3.
    No local file saving — goes straight to S3 bucket.
    Returns session_id and S3 URL for tracking.
    """
    from s3_utils import upload_bytes_to_s3
    from database import save_recording_url
    
    # Generate session_id if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Generate candidate_uuid if not provided
    if not candidate_uuid:
        candidate_uuid = str(uuid.uuid4())
    
    # Read file content into memory
    content = await file.read()
    
    # Build filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = ".webm"
    filename = f"{session_id}_{timestamp}{extension}"
    
    logger.info(f"📤 Uploading {recording_type} recording directly to S3: {filename} ({len(content)} bytes)")
    
    # Upload DIRECTLY to S3 (no local file)
    s3_url = None
    try:
        s3_url = upload_bytes_to_s3(
            file_bytes=content,
            filename=filename,
            session_id=session_id,
            candidate_uuid=candidate_uuid,
            file_type=recording_type
        )
        
        if s3_url:
            # Save S3 URL to MongoDB
            save_recording_url(
                session_id=session_id,
                candidate_uuid=candidate_uuid,
                file_type=recording_type,
                s3_url=s3_url,
                local_path=""  # No local file
            )
            logger.info(f"✅ Direct S3 upload + DB saved: {s3_url}")
        else:
            logger.error(f"❌ S3 upload returned None")
    except Exception as e:
        logger.error(f"❌ S3 upload error: {e}")
    
    return {
        "status": "success" if s3_url else "error",
        "session_id": session_id,
        "candidate_uuid": candidate_uuid,
        "filename": filename,
        "s3_url": s3_url,
        "size_bytes": len(content),
        "recording_type": recording_type
    }



@app.post("/api/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file using Vertex AI Gemini 2.5 Flash.
    Returns text file with AI/User separated transcription with timestamps.
    """
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save uploaded audio temporarily
    temp_audio_path = AUDIO_DIR / f"temp_{session_id}.webm"
    content = await file.read()
    with open(temp_audio_path, "wb") as f:
        f.write(content)
    
    try:
        # Use Vertex AI Gemini 2.5 Flash for transcription
        model = GenerativeModel('gemini-2.5-flash')
        
        # Read audio file and create Part
        with open(temp_audio_path, "rb") as f:
            audio_data = f.read()
        audio_part = Part.from_data(audio_data, mime_type="audio/webm")
        
        # Request transcription with speaker separation
        prompt = """
        Transcribe this audio interview conversation.
        
        IMPORTANT: Separate the speakers as follows:
        - AI Interviewer: The AI voice asking questions
        - User: The human candidate answering questions
        
        Format each line as:
        [TIMESTAMP] SPEAKER: Text
        
        Example:
        [00:00:05] AI: Good afternoon, could you please introduce yourself?
        [00:00:12] User: Yes, my name is John and I have 5 years of experience.
        
        Provide accurate timestamps in MM:SS format.
        Transcribe the COMPLETE conversation.
        """
        
        response = model.generate_content([audio_part, prompt])
        transcript_text = response.text
        
        # Save transcript to file
        transcript_filename = f"transcript_{session_id}_{timestamp}.txt"
        transcript_path = TRANSCRIPTS_DIR / transcript_filename
        
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(f"# Interview Transcript\n")
            f.write(f"# Session ID: {session_id}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
            f.write(transcript_text)
        
        # Extract user-only transcript for scoring
        user_lines = []
        for line in transcript_text.split('\n'):
            if 'User:' in line or 'USER:' in line or 'user:' in line:
                user_lines.append(line)
        
        user_transcript = '\n'.join(user_lines)
        
        # Cleanup temp file
        temp_audio_path.unlink(missing_ok=True)
        
        logger.info(f"Transcription complete: {transcript_filename}")
        
        return {
            "status": "success",
            "session_id": session_id,
            "transcript_file": transcript_filename,
            "transcript_path": str(transcript_path),
            "full_transcript": transcript_text,
            "user_transcript": user_transcript
        }
        
    except Exception as e:
        temp_audio_path.unlink(missing_ok=True)
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/score-communication")
async def score_communication(file: UploadFile = File(...)):
    """
    Analyze user's audio for communication skills using Vertex AI.
    Scores: Pitch, Calmness, Fluency, Confidence, Clarity (0-10 total)
    """
    session_id = str(uuid.uuid4())
    
    # Save uploaded audio temporarily
    temp_audio_path = AUDIO_DIR / f"temp_comm_{session_id}.webm"
    content = await file.read()
    with open(temp_audio_path, "wb") as f:
        f.write(content)
    
    try:
        model = GenerativeModel('gemini-2.5-flash')
        
        # Read audio and create Part
        with open(temp_audio_path, "rb") as f:
            audio_data = f.read()
        audio_part = Part.from_data(audio_data, mime_type="audio/webm")
        
        prompt = """
        Analyze this interview audio for the USER/CANDIDATE's communication skills ONLY.
        Ignore the AI interviewer's voice - focus only on the human candidate.
        
        Score each category from 0-2 points:
        
        1. PITCH (0-2): Is the voice pitch appropriate, not too monotone or too varied?
        2. CALMNESS (0-2): How calm and composed does the candidate sound?
        3. FLUENCY (0-2): How smooth is the speech flow? Minimal filler words (um, uh)?
        4. CONFIDENCE (0-2): Does the candidate sound confident and assured?
        5. CLARITY (0-2): How clear and understandable is the speech?
        
        Respond in this EXACT JSON format:
        {
            "pitch": {"score": X, "feedback": "..."},
            "calmness": {"score": X, "feedback": "..."},
            "fluency": {"score": X, "feedback": "..."},
            "confidence": {"score": X, "feedback": "..."},
            "clarity": {"score": X, "feedback": "..."},
            "total_score": X,
            "overall_feedback": "..."
        }
        
        Be strict but fair in scoring.
        """
        
        response = model.generate_content([audio_part, prompt])
        
        # Parse JSON response
        response_text = response.text
        # Clean up response if wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        score_data = json.loads(response_text.strip())
        
        # Cleanup
        temp_audio_path.unlink(missing_ok=True)
        
        logger.info(f"Communication score: {score_data.get('total_score', 'N/A')}/10")
        
        return {
            "status": "success",
            "session_id": session_id,
            "score_type": "communication",
            "scores": score_data
        }
        
    except json.JSONDecodeError as e:
        temp_audio_path.unlink(missing_ok=True)
        logger.error(f"JSON parse error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse score response")
    except Exception as e:
        temp_audio_path.unlink(missing_ok=True)
        logger.error(f"Communication scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/api/score-technical")
async def score_technical(file: UploadFile = File(...)):
    """
    Analyze transcript text for technical skills using Vertex AI.
    Scores: Technical accuracy, Problem-solving, Relevance (0-10 total)
    """
    session_id = str(uuid.uuid4())
    
    # Read transcript text file
    content = await file.read()
    transcript_text = content.decode("utf-8")
    
    try:
        model = GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Analyze this interview transcript for the CANDIDATE's technical skills.
        Focus ONLY on the User/Candidate responses, not the AI questions.
        
        TRANSCRIPT:
        {transcript_text}
        
        Score each category:
        
        1. TECHNICAL ACCURACY (0-4): Are the technical answers correct and precise?
        2. PROBLEM SOLVING (0-3): Does the candidate show good problem-solving approach?
        3. RELEVANCE (0-3): Are answers relevant to the questions asked?
        
        Respond in this EXACT JSON format:
        {{
            "technical_accuracy": {{"score": X, "feedback": "..."}},
            "problem_solving": {{"score": X, "feedback": "..."}},
            "relevance": {{"score": X, "feedback": "..."}},
            "total_score": X,
            "overall_feedback": "...",
            "strengths": ["...", "..."],
            "areas_to_improve": ["...", "..."]
        }}
        
        Be strict but fair. Score based on actual technical content.
        """
        
        response = model.generate_content(prompt)
        
        # Parse JSON response
        response_text = response.text
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]
        
        score_data = json.loads(response_text.strip())
        
        logger.info(f"Technical score: {score_data.get('total_score', 'N/A')}/10")
        
        return {
            "status": "success",
            "session_id": session_id,
            "score_type": "technical",
            "scores": score_data
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to parse score response")
    except Exception as e:
        logger.error(f"Technical scoring error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.get("/api/recordings/{session_id}")
async def get_recording(session_id: str, recording_type: str = "audio"):
    """
    Get recording file by session ID.
    """
    if recording_type == "screen":
        search_dir = SCREEN_DIR
    else:
        search_dir = AUDIO_DIR
    
    # Find file matching session_id
    for file_path in search_dir.iterdir():
        if session_id in file_path.name:
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type="audio/webm" if recording_type == "audio" else "video/webm"
            )
    
    raise HTTPException(status_code=404, detail="Recording not found")


@app.get("/api/transcript/{session_id}")
async def get_transcript(session_id: str):
    """
    Get transcript file by session ID.
    """
    for file_path in TRANSCRIPTS_DIR.iterdir():
        if session_id in file_path.name:
            return FileResponse(
                path=file_path,
                filename=file_path.name,
                media_type="text/plain"
            )
    
    raise HTTPException(status_code=404, detail="Transcript not found")


# Serve Frontend static files (MUST be after all route definitions)
FRONTEND_DIR = Path(__file__).parent.parent / "Frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1,
        limit_concurrency=MAX_CONCURRENT_CONNECTIONS + 50,  # Buffer for safety
        timeout_keep_alive=75,
        ws_ping_interval=20,
        ws_ping_timeout=10
    )