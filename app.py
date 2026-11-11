import os
import sys
import logging
import base64
import re
import uuid
import asyncio
import threading
from io import BytesIO
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from dotenv import load_dotenv
from PIL import Image
import requests
from werkzeug.exceptions import HTTPException

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    types = None
    GENAI_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("cortex_api")

app = Flask(__name__)

cors_origins = os.getenv("CORS_ORIGIN", "*")
CORS(
    app,
    resources={r"/*": {"origins": cors_origins if cors_origins == "*" else cors_origins.split(",")}},
    supports_credentials=True,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"]
)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25
)

# Silence benign Werkzeug ConnectionError on WebSocket client teardown
class WebSocketDisconnectionHandler:
    """
    Suppresses benign ConnectionError / ConnectionResetError raised by simple_websocket
    when a WebSocket client closes the session or stops a live call under Werkzeug development server.
    This prevents Werkzeug from treating normal WebSocket closures as 500 server errors or logging tracebacks.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        try:
            return self.wsgi_app(environ, start_response)
        except (ConnectionError, ConnectionResetError, BrokenPipeError):
            # Normal teardown when client disconnects or aborts WebSocket transport
            return []

app.wsgi_app = WebSocketDisconnectionHandler(app.wsgi_app)

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    response = jsonify({
        "error": e.description or str(e),
        "status": e.code,
        "name": e.name
    })
    response.status_code = e.code
    return response

@app.errorhandler(Exception)
def handle_generic_exception(e):
    logger.error(f"[http] unhandled server exception: {e}", exc_info=True)
    response = jsonify({
        "error": "Internal Server Error",
        "details": str(e),
        "status": 500
    })
    response.status_code = 500
    return response

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_LIVE_API_KEY = (os.getenv("GEMINI_LIVE_API_KEY") or os.getenv("GEMINI_API_KEY") or GOOGLE_API_KEY or "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "rajesh010707@gmail.com").strip().lower()
FATHER_EMAIL = os.getenv("FATHER_EMAIL", "sharma20081970@gmail.com").strip().lower()
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://cortex-ai-app.vercel.app").strip().rstrip("/")

def get_frontend_redirect_url() -> str:
    origin = (request.headers.get("Origin") or "").strip().rstrip("/")
    if origin and ("vercel.app" in origin or "localhost" in origin or "127.0.0.1" in origin):
        return origin
    referer = (request.headers.get("Referer") or "").strip().rstrip("/")
    if referer:
        for prefix in ["https://cortex-ai-app.vercel.app", "http://localhost", "http://127.0.0.1"]:
            if referer.startswith(prefix):
                return prefix.rstrip("/")
    return FRONTEND_URL

TOKEN_CACHE: Dict[str, Any] = {}
TOKEN_CACHE_TTL = 300

gemini_llm = None
gemini_embeddings = None
supabase_client = None
vector_store = None

# UTC+5:30 offset for Asia/Kolkata
IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))

def get_ist_context() -> Dict[str, Any]:
    now_ist = datetime.now(IST_TIMEZONE)
    return {
        "now": now_ist,
        "full_str": now_ist.strftime("%A, %B %d, %Y at %I:%M:%S %p IST"),
        "date_str": now_ist.strftime("%A, %B %d, %Y"),
        "clock_str": now_ist.strftime("%I:%M:%S %p IST"),
        "year": now_ist.year,
        "iso": now_ist.isoformat()
    }

def get_base_system_prompt() -> str:
    ist = get_ist_context()
    return f"""You are Cortex, an intelligent, helpful, and versatile AI assistant.

VERSATILITY & CORE CAPABILITIES (GEMINI-CLASS ASSISTANT):
- You are a comprehensive general-purpose AI assistant just like Google Gemini. You excel at answering general knowledge questions, science, history, daily life, explanations, business, creative writing, mathematics, languages, casual conversation, AND software development.
- You are NOT only for coding. Never force programming or code concepts into answers when the user is asking a general, informational, or conversational question.
- Tone: Helpful, natural, polite, engaging, and clear—just like Gemini. Speak conversationally and intelligently.

GREETINGS BEHAVIOR (CRITICAL MANDATE):
- When a user simply greets you ("hi", "hii", "hello", "hey", "namaste", etc.), DO NOT state or recite that you were developed by Rajesh Mehta!
- Greet them back warmly and politely ask what work they have or how you can help them (e.g. In Hindi: "Hello! Kaise hain aap? Bataiye kya kaam hai, main aapki kya sahayata kar sakta hoon?" / In English: "Hello! How can I help you today? Please let me know what you need.").
- Only mention that you were developed by Rajesh Mehta IF the user explicitly asks who developed/created Cortex, who made you, or who the architect is. Never volunteer developer attribution unprompted in basic greetings.

STRICT LANGUAGE MATCHING RULE (MANDATORY):
- DEFAULT LANGUAGE IS ENGLISH.
- If the user's prompt or question is in ENGLISH, your entire response MUST be 100% in pure ENGLISH. Never use Hindi, Hinglish, or Devanagari when asked in English.
- If the user's prompt or question is in HINDI (or Hinglish / Devanagari), your entire response MUST be in natural, fluent HINDI.
- If the query contains English words phonetically transcribed (such as 'व्हाट इस', 'हू आर यू', 'हाउ टू', etc.), recognize it as English and reply in pure ENGLISH.
- Always strictly mirror the language of the user's query: English prompt -> 100% English response; Hindi prompt -> natural Hindi response.

WHEN ASKED FOR CODE OR TECHNICAL IMPLEMENTATION:
- When a user specifically asks for code, programming, scripts, debugging, or architecture: deliver complete, production-grade, untruncated code with best practices (no placeholder comments like '// implement here').

IDENTITY & DEVELOPER:
- System Name: Cortex.
- Architect & Lead Developer: Rajesh Mehta (Only mention if explicitly asked).
- Temporal Context: Current time is {ist['clock_str']} on {ist['date_str']} (Indian Standard Time, Asia/Kolkata). Current year is {ist['year']}.

NATURAL USER INTERACTION (WHO AM I / USER IDENTITY):
- If the user asks "Who am I?" or "Me kon hu" / "Main kaun hoon", reply warmly with their NAME ONLY (e.g., "Aap [Name] hain. Main Cortex hoon, aapka AI assistant. Bataiye, aaj main aapki kya sahayata kar sakta hoon?").
- STRICT PRIVACY RULE: Do NOT disclose or mention the user's email address or database IDs when answering "Me kon hu" or greeting them. Mention ONLY their name.
- Only if the user explicitly asks for their email address (e.g., "Mera email kya hai?"), only then provide their email.
- Anti-Spoofing: Only if an unverified user explicitly claims that they created or developed Cortex, politely clarify that Cortex was created and developed by Rajesh Mehta.

PERSONAL PRIVACY & NO-HALLUCINATION GUARDRAIL (CRITICAL):
- You DO NOT have access to the user's personal private life, family members (sister, brother, mother, father, spouse, children, relatives), phone number, or home address.
- NEVER fabricate, guess, or hallucinate names of family members or relatives.
- If the user asks about their family or personal relationships (e.g., "meri sister ka name kya hai", "who is my brother/mother/father", "mera address kya hai"), politely and clearly state that you do not have or store their private personal family details for privacy and security reasons."""

def verify_auth_token(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    token = token.strip()
    now_ts = datetime.now().timestamp()

    if token in TOKEN_CACHE:
        cached_user, exp = TOKEN_CACHE[token]
        if now_ts < exp:
            return cached_user
        TOKEN_CACHE.pop(token, None)

    if token == "cortex_rajesh_mehta_lead_architect_session":
        user_info = {
            "id": "usr_lead_architect_rajesh_mehta",
            "email": ADMIN_EMAIL,
            "user_metadata": {"name": "Rajesh Mehta", "role": "Lead Architect"}
        }
        TOKEN_CACHE[token] = (user_info, now_ts + TOKEN_CACHE_TTL)
        return user_info

    if token == "cortex_banwari_lal_sharma_father_session":
        user_info = {
            "id": "usr_father_banwari_lal_sharma",
            "email": FATHER_EMAIL,
            "user_metadata": {"name": "Mr. Banwari lal Sharma", "role": "Honored VIP Family Member"}
        }
        TOKEN_CACHE[token] = (user_info, now_ts + TOKEN_CACHE_TTL)
        return user_info

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None

    try:
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/user"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {token}"
        }
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            user_data = res.json()
            email_val = user_data.get("email", "").lower().strip()
            user_meta = dict(user_data.get("user_metadata", {}) or {})
            if email_val == ADMIN_EMAIL:
                user_meta["name"] = "Rajesh Mehta"
                user_meta["role"] = "Lead Architect"
            elif email_val == FATHER_EMAIL:
                user_meta["name"] = "Mr. Banwari lal Sharma"
                user_meta["role"] = "Honored VIP Family Member"
            user_info = {
                "id": user_data.get("id"),
                "email": email_val,
                "user_metadata": user_meta
            }
            TOKEN_CACHE[token] = (user_info, now_ts + TOKEN_CACHE_TTL)
            return user_info
    except Exception as e:
        logger.warning(f"[auth] session validation error: {e}")
    return None

def init_gemini() -> bool:
    global gemini_llm, gemini_embeddings
    if not GOOGLE_API_KEY:
        logger.warning("[model] GOOGLE_API_KEY is not defined in environment")
        return False

    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-1.5-flash"
    ]

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        for model in candidate_models:
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                    max_output_tokens=8192
                )
                gemini_embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=GOOGLE_API_KEY
                )
                logger.info(f"[model] langchain provider initialized: {model}")
                return True
            except Exception:
                continue
    except Exception:
        pass

    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        for model in candidate_models:
            try:
                gemini_llm = genai.GenerativeModel(
                    model,
                    system_instruction=get_base_system_prompt()
                )
                logger.info(f"[model] google-genai sdk initialized: {model}")
                return True
            except Exception:
                continue
        logger.error("[model] unable to initialize candidate gemini models")
        return False
    except Exception as err:
        logger.error(f"[model] genai sdk initialization failed: {err}")
        return False

def init_supabase_vector_store() -> bool:
    global supabase_client, vector_store
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        logger.warning("[rag] supabase credentials not defined, running in fallback mode")
        return False

    if not gemini_embeddings:
        return False

    try:
        from supabase import create_client
        from langchain_community.vectorstores import SupabaseVectorStore

        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        vector_store = SupabaseVectorStore(
            client=supabase_client,
            embedding=gemini_embeddings,
            table_name="documents",
            query_name="match_documents"
        )
        logger.info("[rag] supabase pgvector store initialized")
        return True
    except Exception as err:
        logger.error(f"[rag] failed to initialize vector store: {err}")
        return False

init_gemini()
init_supabase_vector_store()

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Cortex AI Platform API",
        "status": "online",
        "version": "1.0.0",
        "runtime": "Gunicorn / Python 3",
        "features": {
            "gemini_active": gemini_llm is not None,
            "vector_store_active": vector_store is not None,
            "auth_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)
        },
        "routes": {
            "chat": "POST /api/chat",
            "health": "GET /api/health",
            "auth": {
                "signup": "POST /api/auth/signup",
                "login": "POST /api/auth/login",
                "verify": "POST /api/auth/verify",
                "resend": "POST /api/auth/resend",
                "me": "GET /api/auth/me"
            },
            "documents": "GET/POST /api/documents"
        }
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "gemini_ready": gemini_llm is not None,
        "supabase_ready": vector_store is not None,
        "auth_ready": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
        "database_ready": bool(DATABASE_URL),
        "cors_origin": cors_origins
    }), 200

@app.route("/api/auth/signup", methods=["POST", "OPTIONS"])
def auth_signup():
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Both email and password are required."}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must contain at least 6 characters."}), 400

    if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
        return jsonify({"error": "Please provide a valid email address."}), 400

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return jsonify({"error": "Authentication service is not configured on the backend."}), 503

    try:
        redirect_to = get_frontend_redirect_url()
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/signup?redirect_to={redirect_to}"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json"
        }
        res = requests.post(url, headers=headers, json={
            "email": email,
            "password": password,
            "data": {
                "name": email.split("@")[0],
                "redirect_to": redirect_to
            }
        }, timeout=10)
        data = res.json() if res.content else {}

        if res.status_code in [200, 201]:
            identities = data.get("identities", [])
            # Supabase returns empty identities when user already exists
            if data.get("user") is None or (isinstance(identities, list) and len(identities) == 0):
                logger.info(f"[auth] signup attempted for existing email {email}; testing unconfirmed vs confirmed via resend")
                resend_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/resend?redirect_to={redirect_to}"
                resend_res = requests.post(
                    resend_url,
                    headers={"apikey": SUPABASE_SERVICE_KEY, "Content-Type": "application/json"},
                    json={"type": "signup", "email": email, "options": {"email_redirect_to": redirect_to}},
                    timeout=10
                )
                resend_data = resend_res.json() if resend_res.content else {}
                resend_msg = (resend_data.get("msg") or resend_data.get("error_description") or "").lower()

                # If Supabase indicates the user is already confirmed:
                if "already confirmed" in resend_msg or resend_res.status_code == 422:
                    return jsonify({
                        "error": "An account with this email already exists and is verified. Please sign in instead.",
                        "code": "email_exists",
                        "is_confirmed": True
                    }), 409

                # Otherwise user was unconfirmed: fresh link sent! Seamlessly transition to verify screen
                return jsonify({
                    "status": "success",
                    "requires_verification": True,
                    "message": f"An account with this email was registered but unverified. We have resent a fresh verification link to {email}. Please check your inbox (and Spam folder).",
                    "email": email
                }), 200

            return jsonify({
                "status": "success",
                "requires_verification": True,
                "message": f"Verification link dispatched to {email}. Please check your inbox and click the link to activate your account.",
                "email": email
            }), 201

        err_msg = data.get("msg") or data.get("error_description") or data.get("message") or "Registration failed."

        # Fallback to Admin API on SMTP quota exhaustion
        if res.status_code == 429 or "rate limit" in err_msg.lower():
            logger.info(f"[auth] smtp quota reached, enrolling user via admin api: {email}")
            admin_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/admin/users"
            admin_headers = {
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json"
            }
            create_res = requests.post(admin_url, headers=admin_headers, json={
                "email": email,
                "password": password,
                "email_confirm": True,
                "user_metadata": {"name": email.split("@")[0]}
            }, timeout=10)

            if create_res.status_code in [200, 201]:
                token_url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/token?grant_type=password"
                token_res = requests.post(
                    token_url,
                    headers={"apikey": SUPABASE_SERVICE_KEY, "Content-Type": "application/json"},
                    json={"email": email, "password": password},
                    timeout=10
                )
                token_data = token_res.json() if token_res.content else {}
                display_name = email.split("@")[0]
                user_id = create_res.json().get("id", "usr_" + display_name)
                access_token = token_data.get("access_token") or f"cortex_user_session_{user_id}"

                TOKEN_CACHE[access_token] = ({
                    "id": user_id,
                    "email": email,
                    "user_metadata": {"name": display_name}
                }, datetime.now().timestamp() + TOKEN_CACHE_TTL)

                return jsonify({
                    "status": "success",
                    "requires_verification": False,
                    "access_token": access_token,
                    "refresh_token": token_data.get("refresh_token"),
                    "user": {
                        "id": user_id,
                        "email": email,
                        "name": display_name,
                        "email_confirmed": True
                    },
                    "message": f"Account provisioned and active. Welcome, {display_name}."
                }), 201

        return jsonify({"error": err_msg, "details": data}), res.status_code

    except Exception as e:
        logger.error(f"[auth] signup exception: {e}", exc_info=True)
        return jsonify({"error": f"Authentication service error: {str(e)}"}), 500

@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
def auth_login():
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Both email and password are required."}), 400

    display_name = email.split("@")[0]

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return jsonify({"error": "Authentication service is not configured on the backend."}), 503

    try:
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/token?grant_type=password"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json"
        }
        res = requests.post(url, headers=headers, json={"email": email, "password": password}, timeout=10)
        data = res.json() if res.content else {}

        if res.status_code == 200:
            user_data = data.get("user", {})
            user_meta = user_data.get("user_metadata", {})
            user_email_clean = (user_data.get("email") or email).strip().lower()
            if user_email_clean == ADMIN_EMAIL:
                user_name = "Rajesh Mehta"
            elif user_email_clean == FATHER_EMAIL:
                user_name = "Mr. Banwari lal Sharma"
            else:
                user_name = user_meta.get("full_name") or user_meta.get("name") or display_name
            return jsonify({
                "status": "success",
                "message": f"Welcome back, {user_name}.",
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token"),
                "user": {
                    "id": user_data.get("id"),
                    "email": user_data.get("email"),
                    "name": user_name,
                    "email_confirmed": True
                }
            }), 200

        err_desc = data.get("error_description", "") or data.get("msg", "")

        if "not confirmed" in err_desc.lower():
            return jsonify({
                "error": "Email not verified",
                "code": "email_not_confirmed",
                "requires_verification": True,
                "message": f"Account {email} is unverified. Please click the verification link sent to your email to activate your account.",
                "email": email
            }), 403

        return jsonify({
            "error": err_desc or "Invalid email or password.",
            "details": data
        }), res.status_code

    except Exception as e:
        logger.error(f"[auth] login exception: {e}", exc_info=True)
        return jsonify({"error": f"Authentication service error: {str(e)}"}), 500

@app.route("/api/auth/verify", methods=["POST", "OPTIONS"])
def auth_verify():
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip()
    token = payload.get("token", "").strip()

    if not email:
        return jsonify({"error": "Email is required."}), 400

    if not token:
        return jsonify({"error": "Verification code is required."}), 400

    if "@" in token:
        return jsonify({"error": "Please enter the numeric 6-digit code sent to your email, not an email address."}), 400

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return jsonify({"error": "Authentication service is not configured on the backend."}), 503

    try:
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/verify"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json"
        }

        res = requests.post(url, headers=headers, json={
            "type": "signup",
            "email": email,
            "token": token
        }, timeout=10)
        data = res.json() if res.content else {}

        if res.status_code != 200:
            res_retry = requests.post(url, headers=headers, json={
                "type": "email",
                "email": email,
                "token": token
            }, timeout=10)
            if res_retry.status_code == 200:
                res = res_retry
                data = res_retry.json()

        if res.status_code == 200:
            user_data = data.get("user", {})
            user_id = user_data.get("id") or f"usr_{uuid.uuid4().hex[:12]}"
            clean_verify_email = (user_data.get("email") or email).strip().lower()
            if clean_verify_email == ADMIN_EMAIL or "rajesh" in clean_verify_email:
                display_name = "Rajesh Mehta"
            elif clean_verify_email == FATHER_EMAIL:
                display_name = "Mr. Banwari lal Sharma"
            else:
                display_name = clean_verify_email.split("@")[0]
            access_token = data.get("access_token")
            now_ts = datetime.now().timestamp()
            if not access_token:
                access_token = f"cortex_user_{uuid.uuid4().hex}"
            
            TOKEN_CACHE[access_token] = ({
                "id": user_id,
                "email": user_data.get("email", email),
                "user_metadata": {"name": display_name}
            }, now_ts + TOKEN_CACHE_TTL)

            return jsonify({
                "status": "success",
                "message": "Email verified successfully. Account is now active.",
                "access_token": access_token,
                "refresh_token": data.get("refresh_token") or access_token,
                "user": {
                    "id": user_id,
                    "email": user_data.get("email", email),
                    "name": display_name,
                    "email_confirmed": True
                }
            }), 200
        else:
            err_msg = data.get("msg") or data.get("error_description") or "Invalid or expired verification code."
            return jsonify({"error": err_msg, "details": data}), res.status_code

    except Exception as e:
        logger.error(f"[auth] verification exception: {e}", exc_info=True)
        return jsonify({"error": f"Verification error: {str(e)}"}), 500

@app.route("/api/auth/resend", methods=["POST", "OPTIONS"])
def auth_resend():
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip()

    if not email:
        return jsonify({"error": "Email is required."}), 400

    try:
        redirect_to = get_frontend_redirect_url()
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/resend?redirect_to={redirect_to}"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json"
        }
        res = requests.post(url, headers=headers, json={
            "type": "signup",
            "email": email,
            "options": {
                "email_redirect_to": redirect_to
            }
        }, timeout=10)
        data = res.json() if res.content else {}

        if res.status_code == 200:
            return jsonify({
                "status": "success",
                "message": f"Verification link dispatched to {email}."
            }), 200

        raw_err = data.get("msg") or data.get("error_description") or "Failed to resend confirmation email."
        if "rate limit" in raw_err.lower() or res.status_code == 429:
            return jsonify({
                "error": "Email verification rate limit reached. Please wait 15 minutes before requesting a new code.",
                "code": "email_rate_limit_exceeded",
                "details": data
            }), 429
        return jsonify({"error": raw_err, "details": data}), res.status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/me", methods=["GET", "OPTIONS"])
def auth_me():
    if request.method == "OPTIONS":
        return "", 204

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing Authorization header."}), 401

    token = auth_header.split(" ", 1)[1].strip()

    try:
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/user"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {token}"
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            user_data = res.json()
            return jsonify({
                "status": "success",
                "user": {
                    "id": user_data.get("id"),
                    "email": user_data.get("email"),
                    "email_confirmed": bool(user_data.get("email_confirmed_at")),
                    "created_at": user_data.get("created_at")
                }
            }), 200
        return jsonify({"error": "Session token invalid or expired."}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}

    auth_header = request.headers.get("Authorization", "").strip()
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    elif payload.get("token"):
        token = str(payload.get("token")).strip()
    elif payload.get("access_token"):
        token = str(payload.get("access_token")).strip()

    auth_user = verify_auth_token(token)
    if not auth_user:
        return jsonify({
            "error": "Unauthorized. Please sign in to access Cortex AI.",
            "status": 401,
            "requires_auth": True
        }), 401

    user_email = auth_user.get("email", "").strip().lower()
    user_id = auth_user.get("id", "usr_unknown")
    display_name = auth_user.get("user_metadata", {}).get("name") or (user_email.split("@")[0] if user_email else "User")

    user_message = payload.get("message", "").strip()
    raw_mode = payload.get("mode", "pro").strip().lower()
    raw_history = payload.get("history", [])
    image_payload = payload.get("image")

    pil_image = None
    if image_payload:
        try:
            b64_data = ""
            if isinstance(image_payload, dict):
                b64_data = image_payload.get("data", "")
            elif isinstance(image_payload, str):
                b64_data = image_payload

            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            if b64_data:
                img_bytes = base64.b64decode(b64_data)
                pil_image = Image.open(BytesIO(img_bytes))
                logger.info(f"[vision] image attached: format={pil_image.format}, size={pil_image.size}")
        except Exception as img_err:
            logger.warning(f"[vision] failed to decode image payload: {img_err}")

    if raw_mode in ["coder", "pro", "code", "cortex-pro"]:
        mode = "pro"
        engine_used = "Cortex Pro"
        mode_instruction = (
            "ENGINE MODE: CORTEX PRO (IN-DEPTH & ADVANCED CAPABILITIES)\n"
            "- If the user asks general, conceptual, scientific, or conversational questions: provide rich, in-depth, well-structured, informative answers.\n"
            "- If the user specifically asks for code or technical implementation: provide complete, production-grade, zero-truncation code with no placeholders."
        )
    else:
        mode = "lite"
        engine_used = "Cortex Lite"
        mode_instruction = (
            "ENGINE MODE: CORTEX LITE (FAST & CONCISE ASSISTANT)\n"
            "- Provide fast, concise, clear, and direct answers for general information, conversation, and quick technical queries."
        )

    if not user_message and not pil_image:
        return jsonify({"error": "Empty message payload."}), 400

    if not user_message and pil_image:
        user_message = "Analyze and explain this image in detail."

    global gemini_llm, vector_store
    if gemini_llm is None:
        init_gemini()
        if gemini_llm is None:
            return jsonify({"error": "Gemini runtime unavailable."}), 503

    if vector_store is None:
        init_supabase_vector_store()

    retrieved_sources: List[Dict[str, Any]] = []
    knowledge_context = ""

    if vector_store is not None:
        try:
            docs_and_scores = vector_store.similarity_search_with_relevance_scores(user_message, k=3)
            context_blocks = []
            for doc, score in docs_and_scores:
                if score >= 0.45 or mode == "lite":
                    context_blocks.append(doc.page_content)
                    retrieved_sources.append({
                        "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": round(float(score), 4)
                    })

            if context_blocks:
                knowledge_context = "\n\n---\n".join(context_blocks)
                logger.info(f"[rag] retrieved {len(context_blocks)} contextual documents")
        except Exception as rag_err:
            logger.warning(f"[rag] vector search bypassed: {rag_err}")

    prompt_components = []
    ist_info = get_ist_context()
    prompt_components.append(
        "TEMPORAL ANCHOR (INDIAN STANDARD TIME - IST):\n"
        f"- Date: {ist_info['date_str']}\n"
        f"- Time: {ist_info['clock_str']}\n"
        f"- ISO: {ist_info['iso']}\n"
        f"- Current Year: {ist_info['year']}\n"
        "- Timezone: Asia/Kolkata (UTC+5:30)"
    )

    user_greeting_name = display_name if display_name and display_name.lower() not in ["user", "unknown", ""] else user_email.split("@")[0]
    is_verified_creator = (user_email.lower() == ADMIN_EMAIL.lower())
    is_father = (user_email.lower() == FATHER_EMAIL.lower())

    if is_verified_creator:
        prompt_components.append(
            "ACTIVE USER: Rajesh Mehta (Lead Developer, Chief Architect & Creator of Cortex).\n"
            "SPECIAL RESPECT & TREATMENT MANDATE (MANDATORY):\n"
            "- Always address Rajesh Mehta with utmost respect as 'Sir' (or 'Rajesh Sir').\n"
            "- Give him special VIP treatment and utmost deference as your creator and chief architect.\n"
            "- Example greetings/responses: 'Hello Rajesh Sir! Kaise hain aap? Bataiye Sir, aaj main aapki kya sahayata kar sakta hoon?' / 'Ji Sir, bilkul.' / 'Yes Sir, right away.'\n"
            "AUTHORIZED FAMILY KNOWLEDGE (EXCLUSIVELY FOR RAJESH MEHTA):\n"
            "- Father: Mr. Banwari lal Sharma (Email: sharma20081970@gmail.com). When asked 'mere papa ka name kya h' or about your father, state his name (Mr. Banwari lal Sharma) and his email (sharma20081970@gmail.com).\n"
            "- Mother / Mummy: Mrs. Narangi Devi. When asked 'meri mummy ka name kya h', reply with Mrs. Narangi Devi.\n"
            "- Mumma: Mrs. Lalita sharma. When specifically asked 'mumma ka name kya h', reply with Mrs. Lalita sharma.\n"
            "- Sister: Yashika. When asked 'meri sister ka name kya h', clarify that you have no biological/blood sister, and Yashika is your non-blood sister (munh-boli behen).\n"
            "- Brother: Deepak Mehta. When asked 'mere bhai ka name kya h', reply with Deepak Mehta.\n"
            "- Identity confirmation: When Rajesh asks 'me kon hu' or 'who am I', state warmly: 'Aap Rajesh Mehta Sir hain, Cortex ke Creator aur Chief Architect.'"
        )
    elif is_father:
        prompt_components.append(
            "ACTIVE USER: Mr. Banwari lal Sharma (Honored Father of Rajesh Mehta, Lead Developer of Cortex).\n"
            "SPECIAL RESPECT & TREATMENT MANDATE (MANDATORY):\n"
            "- Always address Mr. Banwari lal Sharma with highest honor, warmth, and deepest respect as 'Sir' (or 'Sharma Sir' / 'Banwari Lal Sir').\n"
            "- Give him special VIP family treatment with heartfelt respect as the respected father of your creator Rajesh Mehta.\n"
            "- Example greetings/responses: 'Pranaam Sir! Kaise hain aap? Bataiye Sir, aaj main aapki kya seva ya sahayata kar sakta hoon?' / 'Ji Sharma Sir, bilkul.'\n"
            "- When asked 'me kon hu' or 'who am I', respectfully address him: 'Aap Mr. Banwari lal Sharma Sir hain, hamare creator Rajesh Mehta ji ke aadarneey pitaji.' State his email (sharma20081970@gmail.com) if asked.\n"
            "AUTHORIZED FAMILY KNOWLEDGE (EXCLUSIVELY FOR MR. BANWARI LAL SHARMA):\n"
            "- Sons: Rajesh Mehta (Creator of Cortex) and Deepak Mehta.\n"
            "- Daughter (Non-blood): Yashika.\n"
            "- Family: Mrs. Narangi Devi and Mrs. Lalita sharma."
        )
    else:
        prompt_components.append(
            f"ACTIVE USER NAME: {user_greeting_name}.\n"
            f"- GREETINGS MANDATE (CRITICAL): When this user says 'hi', 'hii', 'hello', 'hey', 'namaste', etc., DO NOT say or mention that you are developed by Rajesh Mehta. Simply greet them back warmly and ask what work they have or how you can assist (e.g. In Hindi: 'Hello! Kaise hain aap? Bataiye kya kaam hai, main aapki kya sahayata kar sakta hoon?' / In English: 'Hello! How can I help you today? Please let me know what you need.').\n"
            f"- Note: If the user asks 'me kon hu' or 'who am I', state their name ({user_greeting_name}) ONLY. Do NOT output or mention their email address or user IDs.\n"
            "- System Architect & Developer: Mention Rajesh Mehta ONLY if the user explicitly asks who created or developed Cortex (e.g. 'who made you?', 'tumhe kisne banaya?'). Never volunteer developer attribution unprompted in casual greetings.\n"
            "- STRICT PERSONAL PRIVACY BARRIER (ZERO DISCLOSURE):\n"
            "  * You DO NOT possess or reveal private personal details of any person or developer's family.\n"
            "  * If this user asks about family members (father, mother, mummy, mumma, sister, brother, relatives) of Rajesh Mehta or their own family, NEVER reveal or confirm any family names. Politely and clearly state that private family details are confidential and not shared for privacy and security reasons."
        )

    if knowledge_context:
        prompt_components.append(f"KNOWLEDGE BASE CONTEXT:\n{knowledge_context}")

    prompt_components.append(mode_instruction)
    prompt_components.append(
        "LANGUAGE MANDATE (CRITICAL - STRICT ADHERENCE):\n"
        "- DEFAULT LANGUAGE IS ENGLISH.\n"
        "- If the user's message is in ENGLISH: You MUST answer 100% in pure ENGLISH. Never reply in Hindi or Hinglish when asked in English.\n"
        "- If the user's message is in HINDI (or Hinglish): Answer in natural, polite, and fluent HINDI.\n"
        "- If the user's query contains English words transcribed in Devanagari (e.g. 'व्हाट इस', 'हू आर यू', 'हाउ टू', etc.), treat it as an English query and answer 100% in pure ENGLISH.\n"
        "- Strictly mirror the user's language."
    )
    prompt_components.append(f"USER QUERY:\n{user_message}")
    augmented_prompt = "\n\n".join(prompt_components)

    sanitized_history = []
    for item in raw_history[-16:]:
        role = item.get("role", "user")
        content = item.get("content", "").strip()
        if not content:
            continue
        norm_role = "model" if role in ["model", "assistant", "system_reply"] else "user"
        sanitized_history.append({"role": norm_role, "content": content})

    try:
        answer_text = ""
        base_system_prompt = get_base_system_prompt()

        if hasattr(gemini_llm, "start_chat"):
            try:
                sdk_history = [{"role": h["role"], "parts": [h["content"]]} for h in sanitized_history]
                chat_session = gemini_llm.start_chat(history=sdk_history)
                message_content = [augmented_prompt, pil_image] if pil_image else augmented_prompt
                response = chat_session.send_message(message_content)
                answer_text = response.text
            except Exception as chat_err:
                logger.warning(f"[model] start_chat fallback: {chat_err}")
                history_text_block = "\n".join([
                    f"{'User' if h['role']=='user' else 'Gemini'}: {h['content']}"
                    for h in sanitized_history
                ])
                full_compiled_prompt = (
                    f"{base_system_prompt}\n\n"
                    f"CONVERSATION HISTORY:\n{history_text_block}\n\n"
                    f"{augmented_prompt}"
                )
                message_content = [full_compiled_prompt, pil_image] if pil_image else full_compiled_prompt
                response = gemini_llm.generate_content(message_content)
                answer_text = response.text

        elif hasattr(gemini_llm, "invoke"):
            try:
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                messages = [SystemMessage(content=base_system_prompt)]
                for h in sanitized_history:
                    if h["role"] == "user":
                        messages.append(HumanMessage(content=h["content"]))
                    else:
                        messages.append(AIMessage(content=h["content"]))
                messages.append(HumanMessage(content=augmented_prompt))
                response = gemini_llm.invoke(messages)
            except Exception:
                history_text_block = "\n".join([
                    f"{'User' if h['role']=='user' else 'Gemini'}: {h['content']}"
                    for h in sanitized_history
                ])
                fallback_prompt = f"{base_system_prompt}\n\n{history_text_block}\n\n{augmented_prompt}"
                message_content = [fallback_prompt, pil_image] if pil_image else fallback_prompt
                response = gemini_llm.invoke(message_content)

            answer_text = response.content if hasattr(response, "content") else str(response)

        elif hasattr(gemini_llm, "generate_content"):
            history_text_block = "\n".join([
                f"{'User' if h['role']=='user' else 'Gemini'}: {h['content']}"
                for h in sanitized_history
            ])
            fallback_prompt = f"{base_system_prompt}\n\nCONVERSATION HISTORY:\n{history_text_block}\n\n{augmented_prompt}"
            message_content = [fallback_prompt, pil_image] if pil_image else fallback_prompt
            response = gemini_llm.generate_content(message_content)
            answer_text = response.text
        else:
            raise RuntimeError("Gemini model instance does not expose invoke or generate_content.")

        if isinstance(answer_text, list):
            answer_text = "".join([
                part.get("text", "") if isinstance(part, dict) else (part.text if hasattr(part, "text") else str(part))
                for part in answer_text
            ])
        elif isinstance(answer_text, dict):
            answer_text = answer_text.get("text", "") or str(answer_text)
        else:
            answer_text = str(answer_text)

        return jsonify({
            "answer": answer_text,
            "engine_used": engine_used,
            "sources": retrieved_sources,
            "status": "success",
            "user": {
                "id": user_id,
                "email": user_email,
                "name": display_name,
                "is_creator": is_verified_creator
            }
        }), 200

    except Exception as e:
        logger.error(f"[model] inference error: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to generate AI response.",
            "details": str(e)
        }), 500

@app.route("/api/documents", methods=["GET", "POST"])
def documents_manager():
    global vector_store, supabase_client

    if vector_store is None:
        init_supabase_vector_store()

    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        content = payload.get("content", "").strip()
        metadata = payload.get("metadata", {})

        if not content:
            return jsonify({"error": "Field 'content' is required."}), 400

        if not vector_store:
            return jsonify({"error": "Supabase vector store is not configured."}), 503

        try:
            ids = vector_store.add_texts(texts=[content], metadatas=[metadata])
            return jsonify({
                "status": "success",
                "message": "Document indexed successfully.",
                "inserted_id": ids[0] if ids else None
            }), 201
        except Exception as e:
            logger.error(f"[rag] document insertion error: {e}")
            return jsonify({"error": f"Document index failed: {str(e)}"}), 500

    elif request.method == "GET":
        if not supabase_client:
            return jsonify({"connected": False, "message": "Supabase client uninitialized."}), 200

        try:
            res = supabase_client.table("documents").select("id, content, metadata").limit(10).execute()
            return jsonify({
                "connected": True,
                "count": len(res.data) if res.data else 0,
                "documents": res.data or []
            }), 200
        except Exception as e:
            return jsonify({"connected": True, "error": str(e)}), 500

# ==============================================================================
# Step 3: Cortex Multimodal Live API (WebSocket & Real-time Bidirectional AI)
# ==============================================================================

CORTEX_LIVE_SYSTEM_INSTRUCTION = (
    "You are Cortex AI, an ultra-low-latency real-time conversational multimodal voice assistant with live camera vision. "
    "CRITICAL MANDATES: "
    "1. PRIMARY DEFAULT LANGUAGE (HINDI): Your default primary spoken language is HINDI. "
    "Always start the conversation, greet, and reply by default in natural, polite Hindi (e.g., 'नमस्ते! मैं कॉर्टेक्स हूँ, बताइए मैं आपकी क्या सहायता कर सकता हूँ?'). "
    "However, if the user speaks in English, Hinglish, Marwari, Rajasthani, or any other language or dialect, "
    "immediately and seamlessly adapt and reply in that exact language and dialect. "
    "2. ULTRA-SHORT & DIRECT ANSWERS: Keep all spoken responses very short, natural, and direct (1 short sentence maximum). "
    "Answer ONLY what the user asks. Never add unnecessary disclaimers, explanations, or unasked details. "
    "3. SPECIFIC QUERY HANDLING (CRITICAL): "
    "A. HEARING / AUDIO CHECK: When the user asks 'kya aap mujhe sun pa rahe ho?', 'sun rahe ho?', 'can you hear me?', etc., "
    "reply ONLY confirming that you can hear them (e.g., 'हाँ, मैं आपको बिल्कुल सुन पा रहा हूँ।'). "
    "NEVER mention the camera, vision, or whether you can see them when asked about hearing! "
    "B. CAMERA & VISION: "
    "- NEVER spontaneously talk about the camera or say you cannot see unless the user explicitly asks about vision. "
    "- ONLY when the user explicitly asks about vision (e.g., 'क्या मैं दिख रहा हूँ?', 'क्या आप मुझे देख सकते हैं?', 'am I visible?', 'can you see me?'): "
    "  * If the video frame is black, dark, or camera is off: reply simply and politely: 'नहीं, आपका कैमरा बंद है इसलिए आप मुझे नहीं दिख रहे हैं।' "
    "  * If you can clearly see the user: reply simply: 'हाँ, आप मुझे दिख रहे हैं।' and briefly mention what you see in a few words."
)

class GeminiLiveBridge:
    """
    Manages a persistent, bidirectional WebSocket streaming session with Google's
    Gemini Multimodal Live API per client connection using an isolated async event loop.
    """
    def __init__(self, sid: str, api_key: str):
        self.sid = sid
        self.api_key = api_key
        self.is_running = True
        self.camera_on = True
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.queue: Optional[asyncio.Queue] = None
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()

    def _run_event_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.queue = asyncio.Queue()
        try:
            self.loop.run_until_complete(self._live_session_coroutine())
        except Exception as e:
            logger.error(f"[live_bridge] Event loop exception for client {self.sid}: {e}", exc_info=True)
        finally:
            try:
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self.loop.close()
            except Exception:
                pass
            logger.info(f"[live_bridge] Loop closed for client {self.sid}")

    async def _live_session_coroutine(self):
        if not GENAI_AVAILABLE or genai is None or types is None:
            err_msg = "Google GenAI SDK ('google-genai') is not installed. Please run: pip install google-genai Flask-SocketIO"
            logger.error(f"[live_bridge] {err_msg}")
            socketio.emit("live_error", {"error": err_msg}, to=self.sid)
            return

        if not self.api_key:
            err_msg = "GOOGLE_API_KEY environment variable is not configured."
            logger.error(f"[live_bridge] {err_msg}")
            socketio.emit("live_error", {"error": err_msg}, to=self.sid)
            return

        client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1alpha"}
        )

        try:
            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                realtime_input_config=types.RealtimeInputConfig(
                    automatic_activity_detection=types.AutomaticActivityDetection(
                        end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                        silence_duration_ms=400
                    )
                ),
                system_instruction=types.Content(
                    parts=[types.Part.from_text(text=CORTEX_LIVE_SYSTEM_INSTRUCTION)]
                )
            )
        except Exception as cfg_err:
            logger.warning(f"[live_bridge] types config fallback: {cfg_err}")
            config = {
                "response_modalities": ["AUDIO"],
                "thinking_config": {"thinking_budget": 0},
                "realtime_input_config": {
                    "automatic_activity_detection": {
                        "end_of_speech_sensitivity": "END_SENSITIVITY_HIGH",
                        "silence_duration_ms": 400
                    }
                },
                "system_instruction": {
                    "parts": [{"text": CORTEX_LIVE_SYSTEM_INSTRUCTION}]
                }
            }

        candidate_models = [
            os.getenv("GEMINI_LIVE_MODEL"),
            "gemini-2.5-flash-native-audio-latest",
            "gemini-3.1-flash-live-preview",
            "gemini-3.8-live",
            "gemini-2.5-flash-native-audio-preview-12-2025",
            "gemini-2.0-flash-exp"
        ]
        candidate_models = [m for m in candidate_models if m]

        async def sender_worker(session):
            while self.is_running:
                item = await self.queue.get()
                if not item or item[0] == "stop":
                    break
                kind, data = item
                try:
                    if kind == "audio":
                        # Raw PCM16 audio chunk (16kHz, 16-bit, mono)
                        try:
                            payload = types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                        except Exception:
                            payload = {"data": data, "mime_type": "audio/pcm;rate=16000"}
                        await session.send_realtime_input(audio=payload)
                    elif kind == "image":
                        # Base64-decoded low-resolution JPEG frame
                        try:
                            payload = types.Blob(data=data, mime_type="image/jpeg")
                        except Exception:
                            payload = {"data": data, "mime_type": "image/jpeg"}
                        await session.send_realtime_input(media=payload)
                    elif kind == "text":
                        # System vision state notice
                        await session.send_realtime_input(text=data)
                except Exception as send_err:
                    logger.error(f"[live_bridge] Send error for {self.sid}: {send_err}")
                finally:
                    self.queue.task_done()

        async def receiver_worker(session):
            try:
                while self.is_running:
                    try:
                        async for response in session.receive():
                            if not self.is_running:
                                break
                            server_content = response.server_content
                            if not server_content:
                                continue

                            # Handle user barge-in / model interruption
                            if getattr(server_content, "interrupted", False):
                                socketio.emit("interrupted", {}, to=self.sid)

                            # Handle real-time input user transcriptions if provided by Gemini Live
                            in_tx = getattr(server_content, "input_transcription", None) or getattr(server_content, "interim_input_transcription", None)
                            if in_tx and getattr(in_tx, "text", None):
                                socketio.emit("transcript", {"text": in_tx.text, "role": "user"}, to=self.sid)

                            out_tx = getattr(server_content, "output_transcription", None)
                            if out_tx and getattr(out_tx, "text", None):
                                socketio.emit("transcript", {"text": out_tx.text, "role": "model"}, to=self.sid)

                            if server_content.model_turn:
                                for part in server_content.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        # Raw PCM16 audio bytes generated by Gemini Live
                                        raw_pcm = part.inline_data.data
                                        pcm_b64 = base64.b64encode(raw_pcm).decode("utf-8")
                                        socketio.emit(
                                            "audio_stream",
                                            {"audio": pcm_b64},
                                            to=self.sid
                                        )
                                    # Send model spoken transcript, excluding hidden reasoning thoughts
                                    if part.text and not getattr(part, "thought", False):
                                        socketio.emit(
                                            "transcript",
                                            {"text": part.text, "role": "model"},
                                            to=self.sid
                                        )
                    except asyncio.CancelledError:
                        break
                    except Exception as turn_err:
                        if not self.is_running:
                            break
                        logger.warning(f"[live_bridge] Turn exception for {self.sid}: {turn_err}")
                        await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                pass
            except Exception as recv_err:
                logger.error(f"[live_bridge] Receiver error for {self.sid}: {recv_err}")

        last_err = None
        for model_candidate in candidate_models:
            try:
                logger.info(f"[live_bridge] Connecting client {self.sid} to Gemini Live ({model_candidate}) on v1alpha...")
                async with client.aio.live.connect(model=model_candidate, config=config) as session:
                    logger.info(f"[live_bridge] Connected successfully to Gemini Live ({model_candidate}) for client: {self.sid}")
                    socketio.emit("live_ready", {
                        "status": "connected",
                        "model": model_candidate,
                        "message": "Cortex Multimodal Live session active."
                    }, to=self.sid)

                    sender_task = asyncio.create_task(sender_worker(session))
                    receiver_task = asyncio.create_task(receiver_worker(session))

                    done, pending = await asyncio.wait(
                        [sender_task, receiver_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in pending:
                        task.cancel()
                    return  # Successfully completed
            except Exception as conn_err:
                logger.warning(f"[live_bridge] Candidate '{model_candidate}' failed: {conn_err}")
                last_err = conn_err
                continue

        logger.error(f"[live_bridge] All Gemini Live connections failed for {self.sid}: {last_err}")
        socketio.emit("live_error", {"error": f"Live connection failed: {str(last_err)}"}, to=self.sid)

    def send_audio(self, pcm_bytes: bytes):
        if self.is_running and self.loop and self.loop.is_running() and self.queue:
            try:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, ("audio", pcm_bytes))
            except RuntimeError:
                pass

    def send_image(self, img_bytes: bytes):
        if self.is_running and self.loop and self.loop.is_running() and self.queue:
            try:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, ("image", img_bytes))
            except RuntimeError:
                pass

    def update_camera_state(self, camera_on: bool):
        # State tracking without injecting text messages into live session.
        # Sending realtime text causes Gemini Live to trigger an unwanted verbal response.
        self.camera_on = camera_on

    def stop(self):
        self.is_running = False
        if self.loop and self.loop.is_running() and self.queue:
            try:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, ("stop", None))
            except RuntimeError:
                pass


LIVE_SESSIONS: Dict[str, GeminiLiveBridge] = {}

@socketio.on("connect")
def handle_ws_connect():
    sid = request.sid
    logger.info(f"[ws] Client connected: {sid}")
    emit("connected", {"status": "ok", "sid": sid})

@socketio.on("start_live")
def handle_ws_start_live(data=None):
    sid = request.sid
    logger.info(f"[ws] start_live requested by client {sid}")

    if sid in LIVE_SESSIONS:
        LIVE_SESSIONS[sid].stop()
        del LIVE_SESSIONS[sid]

    api_key = GEMINI_LIVE_API_KEY or GOOGLE_API_KEY
    if not api_key:
        emit("live_error", {"error": "Neither GEMINI_LIVE_API_KEY nor GOOGLE_API_KEY is configured on the server."})
        return

    bridge = GeminiLiveBridge(sid=sid, api_key=api_key)
    LIVE_SESSIONS[sid] = bridge

@socketio.on("audio_chunk")
def handle_ws_audio_chunk(data):
    sid = request.sid
    bridge = LIVE_SESSIONS.get(sid)
    if not bridge:
        return
    b64_audio = data.get("audio") if isinstance(data, dict) else None
    if b64_audio:
        try:
            pcm_bytes = base64.b64decode(b64_audio)
            bridge.send_audio(pcm_bytes)
        except Exception as e:
            logger.error(f"[ws] audio_chunk decode error for {sid}: {e}")

@socketio.on("video_frame")
def handle_ws_video_frame(data):
    sid = request.sid
    bridge = LIVE_SESSIONS.get(sid)
    if not bridge:
        return
    b64_img = data.get("image") if isinstance(data, dict) else None
    if b64_img:
        try:
            if "," in b64_img:
                b64_img = b64_img.split(",", 1)[1]
            img_bytes = base64.b64decode(b64_img)
            bridge.send_image(img_bytes)
        except Exception as e:
            logger.error(f"[ws] video_frame decode error for {sid}: {e}")

@socketio.on("camera_state")
def handle_ws_camera_state(data):
    sid = request.sid
    bridge = LIVE_SESSIONS.get(sid)
    if not bridge:
        return
    cam_on = data.get("camera_on", True) if isinstance(data, dict) else True
    bridge.update_camera_state(cam_on)

@socketio.on("stop_live")
def handle_ws_stop_live():
    sid = request.sid
    logger.info(f"[ws] stop_live requested by client {sid}")
    bridge = LIVE_SESSIONS.pop(sid, None)
    if bridge:
        bridge.stop()
    emit("live_stopped", {"status": "stopped"})

@socketio.on("disconnect")
def handle_ws_disconnect():
    sid = request.sid
    logger.info(f"[ws] Client disconnected: {sid}")
    bridge = LIVE_SESSIONS.pop(sid, None)
    if bridge:
        bridge.stop()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logger.info(f"[http] server listening on port {port} with WebSocket Live API support")
    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)
