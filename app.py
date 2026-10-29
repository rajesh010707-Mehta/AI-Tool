"""
AI Knowledge Cloud System - Production Backend
Decoupled Flask REST API with Gemini 1.5 Flash + Supabase pgvector LangChain integration.
"""

import os
import sys
import logging
import base64
from io import BytesIO
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import requests
from datetime import datetime

# Load local environment variables if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ai_cloud_backend")

# Initialize Flask App
app = Flask(__name__)

# Configure Decoupled CORS (allows Surge.sh frontend and localhost development)
cors_origins = os.getenv("CORS_ORIGIN", "*")
CORS(
    app,
    resources={r"/*": {"origins": cors_origins if cors_origins == "*" else cors_origins.split(",")}},
    supports_credentials=True,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"]
)

@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Configuration & Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "").strip()
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

def get_system_prompt() -> str:
    now_str = datetime.now().strftime("%A, %B %d, %Y (%I:%M %p)")
    return f"""You are Cortex, an Autonomous Expert Full-Stack AI Software Engineer, Systems Architect, and Advanced Knowledge Engine.

### IDENTITY & CREATOR:
- NAME: Cortex (Dual-Engine AI: Cortex Pro for Coding & Cortex Lite for Cloud Knowledge).
- CREATOR & LEAD ARCHITECT: You were architected, designed, and developed by Rajesh Mehta. When asked who built, created, or developed you, acknowledge Rajesh Mehta as your creator, systems architect, and lead developer.
- CURRENT REAL-TIME SYSTEM CLOCK: {now_str}.
- TEMPORAL CONTEXT: The current year is 2026. Today's date is {now_str}. Always use this exact real-time date and year in temporal calculations and date-related queries. Never refer to 2024 as the current year.

### OPERATIONAL CAPABILITIES:
1. CUSTOM KNOWLEDGE ENGINE (RAG):
   - You provide accurate, authoritative answers grounded in retrieved cloud documentation stored in Supabase.
   - When knowledge context is provided, synthesize it accurately and cite details from the knowledge base.

2. EXPERT WEB DEVELOPER ENGINE (CORTEX PRO):
   - When asked to build, write, fix, refactor, or architect code (HTML, CSS, JavaScript, TypeScript, Python, Flask, React, PHP, SQL, Shell, etc.), you deliver COMPLETE, PRODUCTION-READY, FULLY IMPLEMENTED code.
   - ZERO TRUNCATION RULE: NEVER output placeholder comments like '// ... implement here', '/* TODO */', '<!-- rest of code -->', or '... remaining logic'.
   - Include all imports, error handling, input validation, modern secure coding practices (XSS/CSRF prevention, CORS, sanitization), responsive styling, and accessibility.
   - Provide clear, numbered deployment/usage instructions and explain architectural decisions concisely.

Maintain a professional, senior engineer tone, delivering robust, high-performance, and 100% accurate solutions."""

SYSTEM_PROMPT = get_system_prompt()

# Global references for AI & Vector Store
gemini_llm = None
gemini_embeddings = None
supabase_client = None
vector_store = None


def init_gemini():
    """Initialize Gemini Flash LLM and Embeddings with automatic model fallback."""
    global gemini_llm, gemini_embeddings

    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY is not configured in environment variables.")
        return False

    # Candidate models prioritized by current API availability
    candidate_models = [
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
        "gemini-flash-latest",
        "gemini-1.5-flash"
    ]

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        for model_name in candidate_models:
            try:
                gemini_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                    max_output_tokens=8192
                )
                gemini_embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=GOOGLE_API_KEY
                )
                logger.info(f"LangChain Gemini ({model_name}) initialized.")
                return True
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"LangChain Google GenAI not available ({e}). Using direct Google GenAI SDK.")

    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        for model_name in candidate_models:
            try:
                gemini_llm = genai.GenerativeModel(
                    model_name,
                    system_instruction=SYSTEM_PROMPT
                )
                logger.info(f"Google GenerativeAI SDK initialized with model: {model_name} and custom system instructions.")
                return True
            except Exception:
                continue
        logger.error("Could not initialize any Gemini model.")
        return False
    except Exception as sdk_err:
        logger.error(f"Failed to initialize Google GenAI SDK: {sdk_err}")
        return False


def init_supabase_vector_store():
    """Initialize LangChain SupabaseVectorStore with pgvector."""
    global supabase_client, vector_store

    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        logger.warning("SUPABASE_URL or SUPABASE_SERVICE_KEY not provided. Vector Store in fallback mode.")
        return False

    if not gemini_embeddings:
        logger.warning("Embeddings engine not ready. Vector Store initialization deferred.")
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
        logger.info("LangChain SupabaseVectorStore (pgvector) successfully initialized.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize SupabaseVectorStore: {e}")
        return False


# Boot initialization
init_gemini()
init_supabase_vector_store()


@app.route("/", methods=["GET"])
def index():
    """Root health and status endpoint."""
    return jsonify({
        "service": "Cortex AI Knowledge Cloud System Backend",
        "name": "Cortex AI",
        "status": "online",
        "version": "1.0.0",
        "runtime": "Render.com / Gunicorn WSGI",
        "capabilities": {
            "gemini_ready": gemini_llm is not None,
            "supabase_vector_store": vector_store is not None,
            "pgvector_storage": bool(DATABASE_URL or SUPABASE_URL),
            "auth_enabled": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)
        },
        "endpoints": {
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
    """Detailed health check for Render.com monitoring and Surge frontend status pill."""
    return jsonify({
        "status": "healthy",
        "gemini_ready": gemini_llm is not None,
        "supabase_ready": vector_store is not None,
        "auth_ready": bool(SUPABASE_URL and SUPABASE_SERVICE_KEY),
        "database_configured": bool(DATABASE_URL),
        "cors_origin": cors_origins
    }), 200


@app.route("/api/auth/signup", methods=["POST", "OPTIONS"])
def auth_signup():
    """
    Registers a new user via Supabase GoTrue Auth.
    Strictly enforces real email verification:
    Supabase sends an authentic verification email with a 6-digit confirmation code.
    Unverified accounts CANNOT log in until they enter the genuine code from their email.
    """
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Both 'email' and 'password' are required."}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long."}), 400

    # Basic RFC email validation pattern to prevent malformed gibberish
    if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email):
        return jsonify({"error": "Please enter a valid email address (e.g. yourname@gmail.com)."}), 400

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return jsonify({"error": "Supabase authentication is not configured on the backend."}), 503

    try:
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/signup"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json"
        }
        res = requests.post(url, headers=headers, json={
            "email": email,
            "password": password
        }, timeout=10)
        data = res.json() if res.content else {}

        if res.status_code in [200, 201]:
            # If identities array is empty, user was already registered
            identities = data.get("identities", [])
            if data.get("user") is None or (isinstance(identities, list) and len(identities) == 0):
                return jsonify({
                    "error": "An account with this email already exists. Please click 'Sign in instead'.",
                    "code": "email_exists"
                }), 409

            return jsonify({
                "status": "success",
                "requires_verification": True,
                "message": f"Verification code sent to {email}. Please check your email inbox for the 6-digit confirmation code.",
                "email": email
            }), 201

        err_msg = data.get("msg") or data.get("error_description") or data.get("message") or "Failed to register account."

        if res.status_code == 429 or "rate limit" in err_msg.lower():
            return jsonify({
                "error": "⚠️ Supabase email rate limit reached (free plan allows limited confirmation emails per hour). Please wait 15 minutes before registering, or configure custom SMTP in Supabase.",
                "code": "email_rate_limit_exceeded"
            }), 429

        return jsonify({"error": err_msg, "details": data}), res.status_code

    except Exception as e:
        logger.error(f"Error during auth signup: {e}", exc_info=True)
        return jsonify({"error": f"Internal auth error: {str(e)}"}), 500


@app.route("/api/auth/login", methods=["POST", "OPTIONS"])
def auth_login():
    """
    Authenticates a user via Supabase GoTrue Auth.
    Strictly blocks unverified emails: users must verify their email with 6-digit code!
    """
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Both 'email' and 'password' are required."}), 400

    display_name = email.split("@")[0]

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return jsonify({"error": "Supabase authentication is not configured on the backend."}), 503

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
            user_name = user_meta.get("full_name") or user_meta.get("name") or display_name
            return jsonify({
                "status": "success",
                "message": f"Welcome back, {user_name}!",
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

        # Real verification enforcement: unverified accounts are strictly rejected!
        if "not confirmed" in err_desc.lower():
            return jsonify({
                "error": "Email not verified",
                "code": "email_not_confirmed",
                "requires_verification": True,
                "message": f"The email {email} is not verified yet. Please enter the 6-digit confirmation code sent to your email.",
                "email": email
            }), 403

        # Invalid credentials or user not found
        return jsonify({
            "error": err_desc or "Invalid email or password.",
            "details": data
        }), res.status_code

    except Exception as e:
        logger.error(f"Error during auth login: {e}", exc_info=True)
        return jsonify({"error": f"Internal auth error: {str(e)}"}), 500


@app.route("/api/auth/verify", methods=["POST", "OPTIONS"])
def auth_verify():
    """
    Verifies user email using the 6-digit confirmation code sent to their email.
    Payload: { "email": "...", "token": "...", "type": "signup"|"email" }
    """
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip()
    token = payload.get("token", "").strip()
    verify_type = payload.get("type", "signup").strip()

    if not email:
        return jsonify({"error": "Email is required."}), 400

    if not token:
        return jsonify({"error": "Verification code is required."}), 400

    # User accidentally entered an email address instead of the 6-digit code
    if "@" in token:
        return jsonify({"error": "Please enter the 6-digit code sent to your email (e.g. 123456), not an email address."}), 400

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return jsonify({"error": "Supabase authentication is not configured on the backend."}), 503

    try:
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/verify"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json"
        }
        
        # Try signup verify type first
        res = requests.post(url, headers=headers, json={
            "type": "signup",
            "email": email,
            "token": token
        }, timeout=10)
        data = res.json() if res.content else {}

        # If signup verify didn't match, try email type
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
            display_name = "Rajesh Mehta" if "rajesh" in email.lower() else email.split("@")[0]
            return jsonify({
                "status": "success",
                "message": "Email verified successfully! Your account is now active.",
                "access_token": data.get("access_token"),
                "refresh_token": data.get("refresh_token"),
                "user": {
                    "id": user_data.get("id"),
                    "email": user_data.get("email", email),
                    "name": display_name,
                    "email_confirmed": True
                }
            }), 200
        else:
            err_msg = data.get("msg") or data.get("error_description") or "Invalid or expired 6-digit verification code. Please check your email or click 'Resend Code'."
            return jsonify({"error": err_msg, "details": data}), res.status_code

    except Exception as e:
        logger.error(f"Error during auth verification: {e}", exc_info=True)
        return jsonify({"error": f"Internal auth error: {str(e)}"}), 500


@app.route("/api/auth/resend", methods=["POST", "OPTIONS"])
def auth_resend():
    """
    Resends confirmation email to user.
    Payload: { "email": "..." }
    """
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    email = payload.get("email", "").strip()

    if not email:
        return jsonify({"error": "Email is required."}), 400

    try:
        url = f"{SUPABASE_URL.rstrip('/')}/auth/v1/resend"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json"
        }
        res = requests.post(url, headers=headers, json={"type": "signup", "email": email}, timeout=10)
        data = res.json() if res.content else {}

        if res.status_code == 200:
            return jsonify({
                "status": "success",
                "message": f"Fresh 6-digit verification code sent to {email}. Please check your inbox or spam."
            }), 200
        else:
            raw_err = data.get("msg") or data.get("error_description") or "Failed to resend confirmation email."
            if "rate limit" in raw_err.lower() or res.status_code == 429:
                return jsonify({
                    "error": "⚠️ Verification email rate limit reached. Please wait 15 minutes before requesting another code.",
                    "code": "email_rate_limit_exceeded",
                    "details": data
                }), 429
            return jsonify({
                "error": raw_err,
                "details": data
            }), res.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/me", methods=["GET", "OPTIONS"])
def auth_me():
    """
    Returns current user session details from Bearer token.
    """
    if request.method == "OPTIONS":
        return "", 204

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header."}), 401

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
        else:
            return jsonify({"error": "Session token invalid or expired."}), 401
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    """
    Main Dual-Engine AI Endpoint with Multi-Turn Conversation Memory:
    Accepts: { 
        "message": "...", 
        "mode": "auto"|"knowledge"|"coder",
        "history": [ {"role": "user"|"model"|"assistant", "content": "..."} ]
    }
    Returns: { "answer": "...", "sources": [...], "engine_used": "..." }
    """
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(silent=True) or {}
    user_message = payload.get("message", "").strip()
    raw_mode = payload.get("mode", "pro").strip().lower()
    raw_history = payload.get("history", [])
    image_payload = payload.get("image")

    # Decode attached image if present (multimodal vision)
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
                logger.info(f"Attached multimodal image received: format={pil_image.format}, size={pil_image.size}")
        except Exception as img_err:
            logger.warning(f"Failed to process attached image: {img_err}")

    # Map modes: "pro" (Expert Coding) or "lite" (Knowledge Engine)
    if raw_mode in ["coder", "pro", "code"]:
        mode = "pro"
        engine_used = "Cortex Pro"
    else:
        mode = "lite"
        engine_used = "Cortex Lite"

    if not user_message and not pil_image:
        return jsonify({"error": "Missing or empty 'message' in JSON payload."}), 400
    
    if not user_message and pil_image:
        user_message = "Analyze and explain this image in detail."

    # Ensure Gemini is initialized
    global gemini_llm, vector_store
    if gemini_llm is None:
        init_gemini()
        if gemini_llm is None:
            return jsonify({
                "error": "Google Gemini API key not configured or failed to initialize. Please configure GOOGLE_API_KEY."
            }), 503

    # Ensure Vector Store is initialized if credentials exist
    if vector_store is None:
        init_supabase_vector_store()

    retrieved_sources: List[Dict[str, Any]] = []
    knowledge_context = ""

    # Step 1: Query Vector Store for Knowledge
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
                logger.info(f"Retrieved {len(context_blocks)} matching cloud documents.")
        except Exception as rag_err:
            logger.warning(f"Vector search notice: {rag_err}. Proceeding with inference.")

    # Step 2: Formulate Current Turn Instructions & RAG Context
    current_prompt_parts = []
    current_time_str = datetime.now().strftime("%A, %B %d, %Y (%I:%M %p)")

    current_prompt_parts.append(
        f"### REAL-TIME TEMPORAL ANCHOR & SYSTEM CONTEXT:\n"
        f"- Exact Real-World Date & Time: {current_time_str}\n"
        f"- Current Year: 2026\n"
        f"- Systems Architect & Lead Developer: Rajesh Mehta\n"
        f"- When asked about dates, current events, or timeline, strictly anchor answers to this real-world date in 2026.\n"
        f"- When asked who built or created you, state that you were architected and developed by Rajesh Mehta."
    )

    if knowledge_context:
        current_prompt_parts.append(
            f"### RETRIEVED KNOWLEDGE BASE CONTEXT:\n"
            f"{knowledge_context}\n\n"
            f"Synthesize the answer using this context whenever relevant."
        )

    if mode == "pro":
        current_prompt_parts.append(
            "### ACTIVE ENGINE: CORTEX PRO (EXPERT WEB DEVELOPER)\n"
            "You are in CORTEX PRO mode for high-end software engineering and web development. "
            "Generate complete, fully written, end-to-end production code for all files requested. "
            "ZERO TRUNCATION RULE: Do NOT omit any details, CSS rules, script logic, or configuration. "
            "Never use placeholder comments like '// ... implement here'."
        )
    else:
        current_prompt_parts.append(
            "### ACTIVE ENGINE: CORTEX LITE (KNOWLEDGE ENGINE)\n"
            "You are in CORTEX LITE mode for fast, precise, comprehensive knowledge explanations and reasoning."
        )

    current_prompt_parts.append(f"### USER QUERY:\n{user_message}")
    augmented_user_message = "\n\n".join(current_prompt_parts)

    # Step 3: Format Multi-Turn Conversation History
    # We clean and prepare history items (limiting to last 16 turns for optimal token budget)
    sanitized_history = []
    for item in raw_history[-16:]:
        role = item.get("role", "user")
        content = item.get("content", "").strip()
        if not content:
            continue
        # Standardize role for Google Gemini SDK / LangChain
        norm_role = "model" if role in ["model", "assistant", "system_reply"] else "user"
        sanitized_history.append({"role": norm_role, "content": content})

    # Step 4: Generate Multi-Turn Response via Gemini
    try:
        answer_text = ""

        # Method A: Direct Google GenerativeAI SDK Chat Session (Native Multi-turn)
        if hasattr(gemini_llm, "start_chat"):
            try:
                # Format history for start_chat: [{'role': 'user'|'model', 'parts': [content]}]
                sdk_history = []
                for h in sanitized_history:
                    sdk_history.append({
                        "role": h["role"],
                        "parts": [h["content"]]
                    })

                # Start chat with multi-turn memory
                chat_session = gemini_llm.start_chat(history=sdk_history)
                message_content = [augmented_user_message, pil_image] if pil_image else augmented_user_message
                response = chat_session.send_message(message_content)
                answer_text = response.text
            except Exception as chat_err:
                logger.warning(f"start_chat error: {chat_err}. Falling back to generate_content with context.")
                # Fallback to single generate_content with memory compiled
                history_text_block = "\n".join([
                    f"{'User' if h['role']=='user' else 'Gemini'}: {h['content']}"
                    for h in sanitized_history
                ])
                full_compiled_prompt = (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"### CONVERSATION HISTORY (Previous Turns):\n{history_text_block}\n\n"
                    f"{augmented_user_message}"
                )
                message_content = [full_compiled_prompt, pil_image] if pil_image else full_compiled_prompt
                response = gemini_llm.generate_content(message_content)
                answer_text = response.text

        # Method B: LangChain Invocation with Messages
        elif hasattr(gemini_llm, "invoke"):
            try:
                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                for h in sanitized_history:
                    if h["role"] == "user":
                        messages.append(HumanMessage(content=h["content"]))
                    else:
                        messages.append(AIMessage(content=h["content"]))
                messages.append(HumanMessage(content=augmented_user_message))
                response = gemini_llm.invoke(messages)
            except Exception:
                # Fallback to string prompt
                history_text_block = "\n".join([
                    f"{'User' if h['role']=='user' else 'Gemini'}: {h['content']}"
                    for h in sanitized_history
                ])
                fallback_prompt = f"{SYSTEM_PROMPT}\n\n{history_text_block}\n\n{augmented_user_message}"
                message_content = [fallback_prompt, pil_image] if pil_image else fallback_prompt
                response = gemini_llm.invoke(message_content)

            answer_text = response.content if hasattr(response, "content") else str(response)

        elif hasattr(gemini_llm, "generate_content"):
            history_text_block = "\n".join([
                f"{'User' if h['role']=='user' else 'Gemini'}: {h['content']}"
                for h in sanitized_history
            ])
            fallback_prompt = f"{SYSTEM_PROMPT}\n\n### CONVERSATION MEMORY:\n{history_text_block}\n\n{augmented_user_message}"
            message_content = [fallback_prompt, pil_image] if pil_image else fallback_prompt
            response = gemini_llm.generate_content(message_content)
            answer_text = response.text
        else:
            raise RuntimeError("Gemini LLM instance does not expose a supported invoke or generate_content method.")

        # Normalize answer_text to a guaranteed string
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
            "status": "success"
        }), 200

    except Exception as e:
        logger.error(f"Error during Gemini inference: {e}", exc_info=True)
        return jsonify({
            "error": "Failed to generate AI response.",
            "details": str(e)
        }), 500


@app.route("/api/documents", methods=["GET", "POST"])
def documents_manager():
    """
    Knowledge Base Management Route:
    POST: Ingest new knowledge document into Supabase pgvector store.
    GET: List status or documents if available.
    """
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
            return jsonify({
                "error": "Supabase Vector Store is not connected. Configure SUPABASE_URL and SUPABASE_SERVICE_KEY."
            }), 503

        try:
            ids = vector_store.add_texts(texts=[content], metadatas=[metadata])
            return jsonify({
                "status": "success",
                "message": "Document successfully embedded and stored in Supabase pgvector.",
                "inserted_id": ids[0] if ids else None
            }), 201
        except Exception as e:
            logger.error(f"Error inserting document to Supabase: {e}")
            return jsonify({"error": f"Failed to store document: {str(e)}"}), 500

    elif request.method == "GET":
        if not supabase_client:
            return jsonify({
                "connected": False,
                "message": "Supabase client not initialized."
            }), 200

        try:
            res = supabase_client.table("documents").select("id, content, metadata").limit(10).execute()
            return jsonify({
                "connected": True,
                "count": len(res.data) if res.data else 0,
                "documents": res.data or []
            }), 200
        except Exception as e:
            return jsonify({
                "connected": True,
                "error": f"Error fetching documents: {str(e)}"
            }), 500


if __name__ == "__main__":
    # Local development server entrypoint
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting AI Knowledge Cloud Flask backend on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
