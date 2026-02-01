# -*- coding: utf-8 -*-
import base64
import streamlit as st
import os
import time
import hashlib
import json
import logging
import re
import speech_recognition as sr
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from functools import wraps
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from supabase import create_client, Client
from legal_scraper.pdf_highlighter import highlight_terms_in_pdf

# PDF Viewer - try to import, use fallback if not available
try:
    from streamlit_pdf_viewer import pdf_viewer
    PDF_VIEWER_AVAILABLE = True
except ImportError:
    PDF_VIEWER_AVAILABLE = False

import fitz  # PyMuPDF - for rendering PDF as images


# ===========================
# RETRY DECORATOR FOR RESILIENCE
# ===========================
def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for automatic retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

THEME_CSS = {
    "Dark": """
    <style>
    /* ===== LEGALLENS PROFESSIONAL UI THEME ===== */
    
    /* Base Theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Header Branding */
    .main-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card Components */
    .feature-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #475569;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.2);
        border-color: #3b82f6;
    }
    
    /* Risk Score Display */
    .risk-score-container {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 2px solid #475569;
    }
    
    .risk-low { border-color: #22c55e; box-shadow: 0 0 30px rgba(34, 197, 94, 0.3); }
    .risk-medium { border-color: #f59e0b; box-shadow: 0 0 30px rgba(245, 158, 11, 0.3); }
    .risk-high { border-color: #ef4444; box-shadow: 0 0 30px rgba(239, 68, 68, 0.3); }
    
    /* Buttons - Primary Action */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(59, 130, 246, 0.5);
    }
    
    /* Secondary Buttons */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        border-radius: 12px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(90deg, #475569 0%, #64748b 100%);
    }
    
    /* Expander Cards */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        border: 1px solid #475569;
        font-weight: 600;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1e293b;
        padding: 0.5rem;
        border-radius: 16px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* Chat Interface */
    .stChatMessage {
        background: #1e293b;
        border-radius: 16px;
        border: 1px solid #334155;
        padding: 1rem;
    }
    
    /* Metrics Display */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #3b82f6;
    }
    
    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1e293b;
        border: 2px dashed #475569;
        border-radius: 16px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
    }
    
    /* Quick Action Cards */
    .quick-action-btn {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #475569;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .quick-action-btn:hover {
        border-color: #3b82f6;
        transform: scale(1.02);
    }
    
    /* PDF Viewer Container */
    .pdf-viewer-container {
        background: #1e293b;
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid #475569;
    }
    
    /* Welcome Hero Section */
    .hero-section {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
        border: 1px solid #475569;
        margin: 2rem 0;
    }
    
    /* Feature List */
    .feature-item {
        display: flex;
        align-items: center;
        padding: 0.75rem;
        background: rgba(59, 130, 246, 0.1);
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Animations */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #475569;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }
    </style>
    """,
}

QUICK_ACTION_PROMPTS = {
    "summary": "Provide a comprehensive summary of this contract, including parties involved, main obligations, and key terms.",
    "risks": "Identify any potential legal risks, red flags, or concerning clauses in this contract. Be specific and cite sections.",
    "dates": "List all important dates, deadlines, and time-sensitive obligations mentioned in this contract.",
}

# ===========================
# BEST-IN-CLASS MODEL CONFIGURATION
# ===========================
# Optimized for your $5 budget - best value models
LLM_MODEL_PRIMARY = "gpt-4o-mini"      # Best value: fast, cheap, capable (200K TPM)
LLM_MODEL_FALLBACK = "gpt-3.5-turbo"   # Fallback if quota issues
LLM_MAX_TOKENS = 800                    # Increased for fuller chat responses
RISK_MAX_TOKENS = 350

# Optional local model (Ollama) for RAG/chat/risk. Leave empty to skip.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")

# Groq Cloud LLM (free tier: 14,400 requests/day) - fast and high quality
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Fast and capable

# Embedding models - using small for cost efficiency
EMBEDDING_MODEL_PRIMARY = "text-embedding-3-small"   # Best value (1M TPM, cheap)
EMBEDDING_MODEL_FALLBACK = "text-embedding-3-small"  # Same - very reliable

# App configuration
MAX_FILE_SIZE_MB = 10
CACHE_TTL_SECONDS = 600

# Terms to highlight in contracts and a short rationale shown under the viewer
HIGHLIGHT_TERMS = [
    {"term": "indemnity", "why": "Indemnity can be one-sided or uncapped"},
    {"term": "limitation of liability", "why": "Check caps and exclusions"},
    {"term": "liability", "why": "Broad liability language may be risky"},
    {"term": "termination", "why": "Notice and termination rights"},
    {"term": "renewal", "why": "Auto-renewal without notice"},
    {"term": "confidential", "why": "Confidentiality scope and carve-outs"},
    {"term": "governing law", "why": "Ensure intended jurisdiction"},
    {"term": "jurisdiction", "why": "Venue/seat selection"},
    {"term": "intellectual property", "why": "IP ownership and license scope"},
    {"term": "data", "why": "Data protection/processing obligations"},
]

# Risk retrieval focus topics to broaden coverage
RISK_TOPICS = [
    "termination notice penalties",
    "liability cap indemnity",
    "payment terms fees late payment",
    "intellectual property license ownership",
    "dispute resolution governing law jurisdiction",
    "confidentiality data protection privacy",
    "force majeure business continuity",
    "exclusivity non-compete non-solicit",
]

# ===========================
# LOGGING
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("LegalLens")

# ===========================
# SECRETS & CONFIGURATION
# ===========================
load_dotenv()

def _fetch_openai_key() -> str:
    """Retrieve OpenAI API key from Streamlit secrets or environment."""
    key = ""
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = os.getenv("OPENAI_API_KEY", "")
    return key.strip() if isinstance(key, str) else ""


def _openai_enabled() -> bool:
    """Return True only when an OpenAI key is configured and explicitly enabled."""
    # Check Streamlit secrets first, then env vars
    try:
        flag = st.secrets.get("USE_OPENAI", "")
    except Exception:
        flag = ""
    if not flag:
        flag = os.getenv("USE_OPENAI", "")
    flag = flag.strip().lower() if isinstance(flag, str) else ""
    if flag in {"0", "false", "no"}:
        return False
    return bool(_fetch_openai_key())


def _require_openai_client() -> OpenAI:
    key = _fetch_openai_key()
    if not key:
        st.error("⚠️ OpenAI is disabled or API Key not found. Configure it in Streamlit secrets or environment variables.")
        st.stop()
    return OpenAI(api_key=key)


openai_client = None

# ===========================
# CUSTOM CSS - LEGAL TECH THEME
# ===========================
def inject_custom_css(theme: str = "Dark") -> None:
    css = THEME_CSS.get(theme, THEME_CSS["Dark"])
    st.markdown(css, unsafe_allow_html=True)


def _show_ephemeral_message(message: str, level: str = "info") -> None:
    """Display a temporary message that disappears after a short delay."""
    toast_fn = getattr(st, "toast", None)
    icon_map = {
        "info": "ℹ️",
        "warning": "⚠️",
        "success": "✅",
        "error": "❌",
    }

    if callable(toast_fn):
        toast_fn(message, icon=icon_map.get(level, "ℹ️"))
        return

    placeholder = st.empty()
    display_fn = getattr(placeholder, level, placeholder.info)
    display_fn(message)
    time.sleep(3)
    placeholder.empty()


def _validate_pdf_upload(uploaded_file) -> bytes:
    """Validate PDF upload size and type, return bytes if valid."""
    if uploaded_file is None:
        raise ValueError("No file uploaded")

    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported")

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File too large. Max {MAX_FILE_SIZE_MB} MB")

    contents = uploaded_file.getvalue()
    if len(contents) == 0:
        raise ValueError("Uploaded file is empty")
    if not contents.startswith(b"%PDF"):
        raise ValueError("Invalid PDF file")
    try:
        pdf_reader = PdfReader(BytesIO(contents))
        if len(pdf_reader.pages) > 200:
            raise ValueError("File too large. Max 200 pages")
    except ValueError:
        raise
    except Exception:
        raise ValueError("Unable to read PDF")

    return contents


def _trigger_rerun() -> None:
    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
    else:
        st.experimental_rerun()


def _fetch_supabase_config() -> tuple[str, str]:
    url = ""
    key = ""

    try:
        url = st.secrets.get("SUPABASE_URL", "")  # type: ignore[attr-defined]
        key = st.secrets.get("SUPABASE_ANON_KEY", "")  # type: ignore[attr-defined]
    except Exception:
        pass

    if not url:
        url = os.getenv("SUPABASE_URL", "")
    if not key:
        key = os.getenv("SUPABASE_ANON_KEY", "")

    if not url or not key:
        st.error("Supabase configuration missing. Please set SUPABASE_URL and SUPABASE_ANON_KEY in secrets or environment variables.")
        st.stop()

    return url, key


def _extract_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes for keyword detection."""
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except Exception:
        return ""


def _detect_highlight_terms(pdf_bytes: bytes):
    text = _extract_text(pdf_bytes).lower()
    hits = []
    for item in HIGHLIGHT_TERMS:
        if item["term"].lower() in text:
            hits.append(item)
    return hits


def _render_pdf_inline(path: Path, show_download_first: bool = True):
    """Render a PDF with multiple fallback methods for maximum compatibility.
    
    Methods tried in order:
    1. streamlit-pdf-viewer component (best quality)
    2. PDF rendered as images (always works)
    3. Object tag fallback
    """
    try:
        data = path.read_bytes()
    except Exception as exc:
        st.error(f"Could not read annotated PDF: {exc}")
        return
    
    # Always show download button prominently first (guaranteed to work)
    if show_download_first:
        st.download_button(
            label="📥 Download Highlighted PDF",
            data=data,
            file_name=path.name,
            mime="application/pdf",
            use_container_width=True,
            type="primary"
        )
    
    # Let user choose preview method
    preview_method = st.radio(
        "Preview Method",
        ["📄 PDF Viewer", "🖼️ Image Preview", "🌐 Browser Embed"],
        horizontal=True,
        help="If one method doesn't work, try another!"
    )
    
    if preview_method == "📄 PDF Viewer":
        # Method 1: streamlit-pdf-viewer (best quality, uses PDF.js internally)
        if PDF_VIEWER_AVAILABLE:
            try:
                pdf_viewer(data, width=700, height=800)
                return
            except Exception as e:
                st.warning(f"PDF viewer failed: {e}. Try Image Preview instead.")
        else:
            st.warning("PDF viewer component not installed. Using Image Preview.")
            preview_method = "🖼️ Image Preview"
    
    if preview_method == "🖼️ Image Preview":
        # Method 2: Render PDF pages as images (always works!)
        _render_pdf_as_images(path)
        return
    
    if preview_method == "🌐 Browser Embed":
        # Method 3: Object tag (may be blocked by browser)
        b64 = base64.b64encode(data).decode()
        html = f'''
        <div class="pdf-viewer-container">
            <object 
                data="data:application/pdf;base64,{b64}" 
                type="application/pdf" 
                width="100%" 
                height="600px"
                style="border-radius: 12px; border: 1px solid #475569;">
                <p style="text-align: center; padding: 2rem; color: #94a3b8;">
                    ⚠️ Browser blocked the PDF preview.<br>
                    Please try "Image Preview" or use the download button.
                </p>
            </object>
        </div>
        '''
        st.markdown(html, unsafe_allow_html=True)
        st.caption("💡 If you see a blank area above, your browser is blocking the embed. Try 'Image Preview' instead.")


def _render_pdf_as_images(pdf_path: Path, max_pages: int = 20):
    """Render PDF pages as images using PyMuPDF - works on ALL browsers."""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        st.caption(f"📄 Showing {min(total_pages, max_pages)} of {total_pages} pages")
        
        # Page navigation
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page_range = st.slider(
                    "Page Range",
                    1, total_pages,
                    (1, min(5, total_pages)),
                    help="Select which pages to display"
                )
            start_page, end_page = page_range[0] - 1, page_range[1]
        else:
            start_page, end_page = 0, 1
        
        # Render selected pages
        for page_num in range(start_page, min(end_page, max_pages)):
            page = doc[page_num]
            # Render at 2x resolution for clarity
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            
            st.markdown(f"**Page {page_num + 1}**")
            st.image(img_bytes, use_container_width=True)
            
            if page_num < end_page - 1:
                st.markdown("---")
        
        doc.close()
        
    except Exception as e:
        st.error(f"Error rendering PDF as images: {e}")


def _get_supabase_redirect_url() -> str:
    redirect = ""
    try:
        redirect = st.secrets.get("SUPABASE_REDIRECT_URL", "")  # type: ignore[attr-defined]
    except Exception:
        pass

    if not redirect:
        redirect = os.getenv("SUPABASE_REDIRECT_URL", "")

    if not redirect:
        redirect = "http://localhost:8501"

    return redirect.rstrip("/")


@st.cache_resource
def _get_supabase_client(url: str, key: str) -> Client:
    return create_client(url, key)


def _extract_attr(obj, attr: str):
    if obj is None:
        return None
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr)
    return None


def _handle_oauth_redirect(supabase: Client) -> None:
    params = st.query_params
    if "code" not in params:
        return

    auth_code = params.get("code", [""])[0]
    if not auth_code:
        return

    try:
        response = supabase.auth.exchange_code_for_session({"auth_code": auth_code})
        session = getattr(response, "session", None)
        user = getattr(response, "user", None)

        access_token = _extract_attr(session, "access_token")
        refresh_token = _extract_attr(session, "refresh_token")
        email = _extract_attr(user, "email") or ""

        if access_token and refresh_token:
            supabase.auth.set_session(access_token, refresh_token)
            st.session_state["supabase_access_token"] = access_token
            st.session_state["supabase_refresh_token"] = refresh_token

        st.session_state["authenticated"] = True
        st.session_state["username"] = email
        st.session_state.pop("login_error", None)
        st.query_params.clear()
        _trigger_rerun()
    except Exception as err:
        st.query_params.clear()
        st.error(f"Google sign-in failed: {err}")


def _store_session_details(supabase: Client, session, user) -> None:
    access_token = _extract_attr(session, "access_token")
    refresh_token = _extract_attr(session, "refresh_token")
    email = _extract_attr(user, "email") or ""

    if access_token and refresh_token:
        supabase.auth.set_session(access_token, refresh_token)
        st.session_state["supabase_access_token"] = access_token
        st.session_state["supabase_refresh_token"] = refresh_token

    st.session_state["authenticated"] = True
    st.session_state["username"] = email
    st.session_state.pop("login_error", None)


def _sync_supabase_session(supabase: Client) -> None:
    access_token = st.session_state.get("supabase_access_token")
    refresh_token = st.session_state.get("supabase_refresh_token")
    if access_token and refresh_token:
        try:
            supabase.auth.set_session(access_token, refresh_token)
        except Exception:
            pass


def _precompute_quick_actions(file_hash: str, vector_store) -> None:
    """Lazy precomputation - now runs only when quick actions are clicked."""
    # Skip automatic precomputation - let buttons trigger on-demand
    # This removes the 5-10 second delay on page load
    cache = st.session_state.get("quick_action_cache", {})
    if cache.get("hash") != file_hash:
        # Reset cache for new document, but don't precompute
        st.session_state["quick_action_cache"] = {"hash": file_hash, "data": {}}


def _render_auth_panel(supabase: Client) -> None:
    st.title("⚖️ LegalLens")
    st.markdown("### Secure Access")
    st.markdown("Sign in with your email to continue.")

    tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
            else:
                try:
                    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    session = getattr(response, "session", None)
                    user = getattr(response, "user", None)

                    if session:
                        _store_session_details(supabase, session, user)
                        _trigger_rerun()
                    else:
                        st.info("Please check your inbox to confirm your email before signing in.")
                except Exception as err:
                    st.error(f"Sign-in failed: {err}")

    with tab_signup:
        with st.form("signup_form", clear_on_submit=False):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Email and password are required.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    response = supabase.auth.sign_up({"email": email, "password": password})
                    session = getattr(response, "session", None)
                    user = getattr(response, "user", None)

                    if session:
                        _store_session_details(supabase, session, user)
                        _trigger_rerun()
                    else:
                        st.success("Account created! Please check your email to confirm before signing in.")
                        st.info("📧 **Can't find the email?** Please check your spam/junk folder.")
                except Exception as err:
                    st.error(f"Sign-up failed: {err}")

    if st.session_state.get("login_error"):
        st.error(st.session_state["login_error"])

    st.caption("Questions about access? Contact your LegalLens administrator.")


def _logout(supabase: Client) -> None:
    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    keys_to_clear = [
        "authenticated",
        "username",
        "login_error",
        "quick_action_output",
        "quick_action_type",
        "quick_action_title",
        "voice_audio_bytes",
        "voice_result",
        "voice_clear_pending",
        "embedding_notice_shown",
        "messages",
        "supabase_access_token",
        "supabase_refresh_token",
        "quick_action_cache",
        "current_file_hash",
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    _trigger_rerun()

# ===========================
# TEXT RAG LOGIC
# ===========================
@st.cache_resource(show_spinner=False)
def _get_openai_embeddings_large() -> OpenAIEmbeddings:
    """Best quality embeddings - 3072 dimensions."""
    if not _openai_enabled():
        raise RuntimeError("OpenAI disabled")
    key = _fetch_openai_key()
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_PRIMARY,  # text-embedding-3-large
        openai_api_key=key,
        chunk_size=500,  # Process in batches for speed
        request_timeout=60,
    )


@st.cache_resource(show_spinner=False)
def _get_openai_embeddings_small() -> OpenAIEmbeddings:
    """Fast fallback embeddings - 1536 dimensions."""
    if not _openai_enabled():
        raise RuntimeError("OpenAI disabled")
    key = _fetch_openai_key()
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_FALLBACK,  # text-embedding-3-small
        openai_api_key=key,
        chunk_size=500,
        request_timeout=60,
    )


@st.cache_resource(show_spinner=False)
def _get_local_embeddings() -> HuggingFaceEmbeddings:
    """Free local fallback - 384 dimensions."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64,
        }
    )


def _get_ollama_model() -> str:
    """Return configured Ollama model or autodetect the first available model."""
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if model:
        logger.info(f"[Ollama] Using env model: {model}")
        return model

    try:
        url = f"{os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/tags"
        logger.info(f"[Ollama] Auto-detecting models from {url}")
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
            models = data.get("models", [])
            logger.info(f"[Ollama] Found models: {[m.get('name') for m in models]}")
            if models:
                detected = models[0].get("name", "").strip()
                logger.info(f"[Ollama] Using detected model: {detected}")
                return detected
    except Exception as e:
        logger.warning(f"[Ollama] Auto-detect failed: {e}")
        return ""

    return ""


def _use_ollama() -> bool:
    return bool(_get_ollama_model())


def _ollama_generate(prompt: str, max_tokens: int, temperature: float = 0.3) -> str:
    model = _get_ollama_model()
    if not model:
        raise RuntimeError("OLLAMA_MODEL not configured and no local models found")

    url = f"{os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    logger.info(f"[Ollama] Calling {url} with model={model}")
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        logger.info(f"[Ollama] Got response ({len(text)} chars)")
        if not text:
            raise RuntimeError("Empty response from Ollama")
        return text


# ===========================
# GROQ CLOUD LLM (FREE & FAST)
# ===========================
def _get_groq_api_key() -> str:
    """Get Groq API key from secrets or environment."""
    key = ""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    if not key:
        key = os.getenv("GROQ_API_KEY", "")
    return key.strip()


def _use_groq() -> bool:
    """Check if Groq is configured and available."""
    return bool(_get_groq_api_key())


def _get_groq_model() -> str:
    """Get configured Groq model."""
    model = ""
    try:
        model = st.secrets.get("GROQ_MODEL", "")
    except Exception:
        pass
    if not model:
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return model.strip()


def _groq_generate(prompt: str, max_tokens: int, temperature: float = 0.3) -> str:
    """Generate text using Groq Cloud API (free, fast)."""
    api_key = _get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not configured")
    
    model = _get_groq_model()
    logger.info(f"[Groq] Calling with model={model}")
    
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    text = response.choices[0].message.content or ""
    logger.info(f"[Groq] Got response ({len(text)} chars)")
    if not text:
        raise RuntimeError("Empty response from Groq")
    return text


@st.cache_resource(show_spinner=False)
def _get_chat_llm_primary() -> ChatOpenAI:
    """Primary LLM - GPT-4o for best quality."""
    if not _openai_enabled():
        raise RuntimeError("OpenAI disabled")
    key = _fetch_openai_key()
    return ChatOpenAI(
        model=LLM_MODEL_PRIMARY,  # gpt-4o
        openai_api_key=key,
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS,
        timeout=30,
        request_timeout=30,
    )


@st.cache_resource(show_spinner=False)
def _get_chat_llm_fallback() -> ChatOpenAI:
    """Fallback LLM - GPT-3.5-turbo for speed/cost."""
    if not _openai_enabled():
        raise RuntimeError("OpenAI disabled")
    key = _fetch_openai_key()
    return ChatOpenAI(
        model=LLM_MODEL_FALLBACK,  # gpt-3.5-turbo
        openai_api_key=key,
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS,
        timeout=20,
        request_timeout=20,
    )


def _get_chat_llm() -> ChatOpenAI:
    """Get best available LLM with automatic fallback."""
    try:
        return _get_chat_llm_primary()
    except Exception:
        return _get_chat_llm_fallback()


@st.cache_resource(show_spinner=False)
def _get_risk_llm_primary() -> ChatOpenAI:
    """Primary Risk LLM - GPT-4o for accurate analysis."""
    if not _openai_enabled():
        raise RuntimeError("OpenAI disabled")
    key = _fetch_openai_key()
    return ChatOpenAI(
        model=LLM_MODEL_PRIMARY,  # gpt-4o
        openai_api_key=key,
        temperature=0.2,
        max_tokens=RISK_MAX_TOKENS,
        timeout=30,
        request_timeout=30,
        response_format={"type": "json_object"},
    )


@st.cache_resource(show_spinner=False)
def _get_risk_llm_fallback() -> ChatOpenAI:
    """Fallback Risk LLM - GPT-3.5-turbo."""
    if not _openai_enabled():
        raise RuntimeError("OpenAI disabled")
    key = _fetch_openai_key()
    return ChatOpenAI(
        model=LLM_MODEL_FALLBACK,
        openai_api_key=key,
        temperature=0.2,
        max_tokens=RISK_MAX_TOKENS,
        timeout=20,
        request_timeout=20,
        response_format={"type": "json_object"},
    )


def _get_risk_llm() -> ChatOpenAI:
    """Get best available Risk LLM with automatic fallback."""
    try:
        return _get_risk_llm_primary()
    except Exception:
        return _get_risk_llm_fallback()


@st.cache_resource(show_spinner=False)
def build_vector_store(file_hash: str, file_bytes: bytes):
    """Extract text, split, and create FAISS vector store with best embeddings."""
    pdf_reader = PdfReader(BytesIO(file_bytes))
    page_count = len(pdf_reader.pages)
    text_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
    text = "\n".join(text_chunks)  # Better separation between pages

    # Semantic-aware chunking with sentence boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,       # Optimal for context window
        chunk_overlap=150,     # Overlap for context continuity
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],  # Semantic breaks
    )
    chunks = text_splitter.split_text(text)

    vector_store = None
    embedding_source = None
    errors = []
    
    # Try embeddings preferring local first to avoid quota issues
    embedding_attempts = [
        ("all-MiniLM-L6-v2 (local)", _get_local_embeddings),
        ("text-embedding-3-small", _get_openai_embeddings_small),
        ("text-embedding-3-large", _get_openai_embeddings_large),
    ]
    
    for source_name, get_embeddings_fn in embedding_attempts:
        try:
            embeddings = get_embeddings_fn()
            vector_store = FAISS.from_texts(chunks, embeddings)
            embedding_source = source_name
            break  # Success!
        except Exception as err:
            errors.append(f"{source_name}: {err}")
            continue
    
    if vector_store is None:
        raise RuntimeError(f"All embeddings failed: {'; '.join(errors)}")

    embedding_details = {
        "embedding_source": embedding_source,
        "chunk_count": len(chunks),
        "fallback_errors": errors[:-1] if len(errors) > 1 else None,
        "file_hash": file_hash,
        "page_count": page_count,
        "file_size_kb": round(len(file_bytes) / 1024, 2),
    }

    return vector_store, text, embedding_details

# Cache for RAG responses - avoid recomputing same questions
@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached_rag_response(file_hash: str, question: str, _vector_store) -> str:
    """Cached wrapper for RAG responses keyed by file hash."""
    return _execute_rag_query(_vector_store, question)


@retry_with_backoff(max_retries=3, base_delay=1.0)
def _execute_rag_query(vector_store, question: str) -> str:
    """Execute RAG query with manual retrieve-then-generate."""

    docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""You are LegalLens, an expert AI legal contract analyzer with deep expertise in contract law.

CONTEXT FROM CONTRACT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided contract context
2. Cite specific sections, clauses, or page references when possible
3. Highlight any legal risks, red flags, or concerning terms
4. If the information is not in the context, clearly state that
5. Use bullet points for clarity when listing multiple items
6. Be precise, professional, and actionable

DETAILED ANSWER:"""

    # Priority: Groq (fast cloud) > Ollama (local) > OpenAI
    if _use_groq():
        try:
            logger.info("[RAG] Using Groq for chat")
            return _groq_generate(prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.3)
        except Exception as e:
            logger.warning(f"[RAG] Groq failed: {e}")

    if _use_ollama():
        try:
            logger.info("[RAG] Using Ollama for chat")
            return _ollama_generate(prompt, max_tokens=LLM_MAX_TOKENS, temperature=0.3)
        except Exception as e:
            logger.warning(f"[RAG] Ollama failed: {e}")

    if not _openai_enabled():
        raise RuntimeError("No LLM configured. Set GROQ_API_KEY, OLLAMA_MODEL, or OPENAI_API_KEY")

    logger.info("[RAG] Using OpenAI for chat")
    llm = _get_chat_llm()
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def get_rag_response(vector_store, question: str) -> str:
    """Generate RAG-based response with caching."""
    file_hash = st.session_state.get("current_file_hash", "default")
    return _cached_rag_response(file_hash, question, vector_store)

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached_risk_score(file_hash: str, _vector_store) -> dict:
    """Cached risk score calculation keyed by file hash."""
    return _execute_risk_calculation(_vector_store)


@retry_with_backoff(max_retries=2, base_delay=1.0)
def _execute_risk_calculation(vector_store) -> dict:
    """Execute comprehensive risk calculation with focused retrieval + structured JSON."""

    def _collect_risk_context(vs):
        snippets = []
        seen = set()
        for topic in RISK_TOPICS:
            for doc in vs.similarity_search(topic, k=2):
                key = hash(doc.page_content)
                if key in seen:
                    continue
                seen.add(key)
                snippets.append(doc.page_content)
        return "\n\n---\n\n".join(snippets), len(snippets)

    def _level_from_score(score_val: int) -> str:
        if score_val <= 30:
            return "Low"
        if score_val <= 60:
            return "Medium"
        return "High"

    def _confidence_level(score: float) -> str:
        if score >= 0.67:
            return "High"
        if score >= 0.34:
            return "Medium"
        return "Low"

    def _normalize_risk_payload(raw_text: str, snippet_count: int) -> dict:
        logger.info(f"[Risk] Raw response text: {raw_text[:500]}...")
        try:
            data = json.loads(raw_text)
            logger.info(f"[Risk] Parsed JSON directly: {list(data.keys())}")
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                logger.info(f"[Risk] Extracted JSON via regex: {match.group()[:200]}...")
                data = json.loads(match.group())
            else:
                logger.warning("[Risk] No JSON found in response")
                data = {}
        except Exception as e:
            logger.error(f"[Risk] JSON parse error: {e}")
            data = {}

        score = data.get("score")
        try:
            score = int(round(float(score)))
        except Exception:
            score = 50
        score = max(0, min(100, score))

        level = data.get("level") or _level_from_score(score)
        top_risks = data.get("top_risks") or data.get("risks") or []
        logger.info(f"[Risk] Parsed top_risks type={type(top_risks).__name__}, value={top_risks}")
        if isinstance(top_risks, dict):
            top_risks = list(top_risks.values())
        if not isinstance(top_risks, list):
            top_risks = [str(top_risks)]

        cleaned_risks = []
        for item in top_risks:
            if isinstance(item, dict):
                issue = item.get("issue") or item.get("risk") or item.get("title") or "Unspecified risk"
                evidence = item.get("evidence") or item.get("clause") or item.get("excerpt")
                severity = item.get("severity") or None
                recommendation = item.get("recommendation")
                cleaned_risks.append(
                    {k: v for k, v in {
                        "issue": issue,
                        "evidence": evidence,
                        "severity": severity,
                        "recommendation": recommendation,
                    }.items() if v}
                )
            else:
                cleaned_risks.append(str(item))

        rationale = data.get("rationale") or data.get("summary")
        confidence_val = data.get("confidence")
        try:
            confidence_val = float(confidence_val)
        except Exception:
            confidence_val = None

        evidence_hits = sum(1 for r in cleaned_risks if isinstance(r, dict) and r.get("evidence"))
        risk_items = len(cleaned_risks) or 1
        evidence_ratio = evidence_hits / risk_items
        coverage_ratio = min(snippet_count / 6, 1.0) if snippet_count else 0.0
        derived_confidence = 0.3 * coverage_ratio + 0.6 * evidence_ratio + 0.1
        derived_confidence = max(0.0, min(1.0, derived_confidence))
        confidence_score = confidence_val if confidence_val is not None else derived_confidence

        return {
            "score": score,
            "level": level,
            "top_risks": cleaned_risks,
            "rationale": rationale,
            "confidence": confidence_score,
            "confidence_level": _confidence_level(confidence_score),
        }

    context, snippet_count = _collect_risk_context(vector_store)

    # Risk analysis prompt for Groq/Ollama
    risk_prompt_local = f"""TASK: You are a legal risk analyst. Analyze this contract excerpt and return a JSON risk assessment.

CONTRACT TEXT:
{context[:2000] if len(context) > 2000 else context}

INSTRUCTIONS:
1. Score the contract risk from 0 (safe) to 100 (very risky)
2. Identify 1-3 specific risk issues with evidence quotes
3. Return ONLY valid JSON in this exact format:

{{"score": 45, "level": "Medium", "rationale": "Brief explanation", "top_risks": [{{"issue": "Risk description", "evidence": "Quote from contract"}}], "confidence": 0.7}}

RESPOND WITH JSON ONLY, NO OTHER TEXT:"""

    # Priority: Groq (fast cloud) > Ollama (local) > OpenAI
    if _use_groq():
        try:
            logger.info("[Risk] Using Groq for risk calculation")
            result_text = _groq_generate(risk_prompt_local, max_tokens=400, temperature=0.1)
            return _normalize_risk_payload(result_text, snippet_count)
        except Exception as e:
            logger.error(f"[Risk] Groq failed: {e}")

    if _use_ollama():
        try:
            logger.info("[Risk] Using Ollama for risk calculation")
            # Use smaller context for local models to avoid timeouts
            short_context = context[:1500] if len(context) > 1500 else context
            ollama_prompt = f"""TASK: You are a legal risk analyst. Analyze this contract excerpt and return a JSON risk assessment.

CONTRACT TEXT:
{short_context}

INSTRUCTIONS:
1. Score the contract risk from 0 (safe) to 100 (very risky)
2. Identify 1-3 specific risk issues with evidence quotes
3. Return ONLY valid JSON in this exact format:

{{"score": 45, "level": "Medium", "rationale": "Brief explanation", "top_risks": [{{"issue": "Risk description", "evidence": "Quote from contract"}}], "confidence": 0.7}}

RESPOND WITH JSON ONLY, NO OTHER TEXT:"""
            result_text = _ollama_generate(ollama_prompt, max_tokens=300, temperature=0.1)
            return _normalize_risk_payload(result_text, snippet_count)
        except Exception as e:
            logger.error(f"[Risk] Ollama failed: {e}")

    if not _openai_enabled():
        return {
            "score": 50,
            "level": "Medium",
            "top_risks": ["No LLM configured. Set GROQ_API_KEY, OLLAMA_MODEL, or OPENAI_API_KEY."],
            "rationale": None,
            "confidence": 0.2,
            "confidence_level": "Low",
        }

    risk_prompt = f"""You are a senior commercial contracts lawyer performing a risk review.

CONTEXT FROM CONTRACT (multiple excerpts):
{context}

TASK:
- Identify the most material risk issues and cite supporting clauses/phrases.
- Score overall risk from 0-100 where 0 = very safe, 100 = very risky.
- Balance probability and impact; reduce confidence when evidence is thin.
- Stay grounded in the provided excerpts. If uncertain, say so.

OUTPUT JSON ONLY:
{{
  "score": <0-100 integer>,
  "level": "Low|Medium|High",
  "rationale": "1-2 sentence overall rationale",
  "top_risks": [
    {{"issue": "...", "evidence": "quoted clause or summary", "severity": "Low|Medium|High", "recommendation": "succinct fix/ask"}},
    ... up to 5 items
    ],
    "confidence": <0-1 float, lower when evidence is thin or excerpts sparse>
}}

GRADING GUARDRAILS:
- High: uncapped indemnity, unlimited liability, aggressive termination, broad IP transfer, one-sided jurisdiction, unclear payment or penalties.
- Medium: capped but narrow liability, moderate notice, ambiguous IP/license scope, limited data protections.
- Low: mutual indemnities, balanced caps, clear payment timing, reasonable jurisdiction, standard confidentiality.
"""

    llm = _get_risk_llm()

    try:
        result = llm.invoke(risk_prompt)
        result_text = getattr(result, "content", str(result))
        return _normalize_risk_payload(result_text, snippet_count)
    except Exception as e:
        return {
            "score": 50,
            "level": "Medium",
            "top_risks": [f"Error generating risk analysis: {str(e)[:80]}"] if str(e) else ["Unable to generate risk analysis"],
            "rationale": None,
            "confidence": 0.2,
            "confidence_level": "Low",
        }


def calculate_risk_score(file_hash: str, vector_store) -> dict:
    """Generate AI-powered contract risk score with caching."""
    return _cached_risk_score(file_hash, vector_store)

# ===========================
# AUDIO LOGIC
# ===========================
@retry_with_backoff(max_retries=2, base_delay=0.5)
def _transcribe_with_openai(audio_bytes: bytes) -> tuple[str, str | None]:
    """OpenAI Whisper transcription - best accuracy."""
    if not _openai_enabled():
        raise RuntimeError("OpenAI disabled")
    client = _require_openai_client()
    audio_buffer = BytesIO(audio_bytes)
    audio_buffer.name = "question.wav"
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_buffer,
        language="en",
        response_format="text",
    )
    return transcription.strip(), "whisper-1"


def _transcribe_with_google_free(audio_bytes: bytes) -> tuple[str, str | None]:
    """Fallback to free Google Speech Recognition."""
    recognizer = sr.Recognizer()
    audio_buffer = BytesIO(audio_bytes)
    
    with sr.AudioFile(audio_buffer) as source:
        audio_data = recognizer.record(source)
    
    text = recognizer.recognize_google(audio_data)
    return text.strip(), "google_free"


def get_audio_response(audio_bytes, vector_store):
    """Transcribe audio with intelligent fallback and answer via RAG."""
    question_text = None
    transcription_source = None
    
    # If OpenAI is disabled, go straight to Google
    if not _openai_enabled():
        try:
            question_text, transcription_source = _transcribe_with_google_free(audio_bytes)
        except Exception as google_err:
            return {
                "error": f"Google Speech Recognition failed: {google_err}",
                "question": None
            }
    else:
        # Try OpenAI Whisper first (best quality)
        try:
            question_text, transcription_source = _transcribe_with_openai(audio_bytes)
        except Exception as openai_err:
            error_str = str(openai_err).lower()
            if any(x in error_str for x in ["429", "quota", "insufficient", "rate_limit", "disabled"]):
                try:
                    question_text, transcription_source = _transcribe_with_google_free(audio_bytes)
                except Exception as google_err:
                    return {
                        "error": f"OpenAI quota exceeded. Google fallback failed: {google_err}",
                        "question": None
                    }
            else:
                return {"error": f"Transcription error: {openai_err}", "question": None}

    if not question_text:
        return {
            "error": "Unable to understand the audio. Please speak clearly and try again.",
            "question": None
        }

    try:
        answer = get_rag_response(vector_store, question_text)
        source_note = " *(transcribed via free Google Speech)* " if transcription_source == "google_free" else ""
        return {"question": question_text + source_note, "answer": answer}
    except Exception as err:
        return {"error": f"Error generating answer: {err}", "question": question_text}

# ===========================
# MAIN APPLICATION
# ===========================
def main():
    inject_custom_css()
    supabase_url, supabase_key = _fetch_supabase_config()
    supabase = _get_supabase_client(supabase_url, supabase_key)

    _sync_supabase_session(supabase)
    _handle_oauth_redirect(supabase)

    if not st.session_state.get("authenticated"):
        _render_auth_panel(supabase)
        return

    username = st.session_state.get("username", "")
    contract_bytes = None
    contract_label = None

    with st.sidebar:
        st.markdown(f"👤 **Signed in as:** `{username}`")
        if st.button("🔓 Log Out", use_container_width=True):
            _logout(supabase)
            st.stop()

        st.markdown("---")
        st.header("📄 Contract Database")
        
        # Option to use default or upload
        use_default = st.checkbox("📚 Use Demo Contract", value=True)
        
        if use_default:
            demo_file = "Contract document.pdf"
            if os.path.exists(demo_file):
                with open(demo_file, "rb") as f:
                    contract_bytes = f.read()
                contract_label = "Contract Document (Demo)"
                st.success("✅ Loaded: Contract Document (Demo)")
                st.info("💡 Uncheck to upload your own contract")
            else:
                st.error(f"❌ Demo file '{demo_file}' not found!")
        else:
            uploaded_file = st.file_uploader(
                "Upload a PDF contract",
                type=["pdf"],
                help="Upload a legal contract in PDF format"
            )
            
            if uploaded_file:
                try:
                    contract_bytes = _validate_pdf_upload(uploaded_file)
                    contract_label = uploaded_file.name
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                    st.info(f"📊 Size: {uploaded_file.size / 1024:.2f} KB")
                except ValueError as ve:
                    st.error(str(ve))
        
        # Suggestion for expanding database
        with st.expander("💡 Expand Your Database"):
            st.markdown("""
            **Suggested Additions:**
            - Employment contracts
            - NDA templates
            - Service agreements
            - Lease agreements
            - Partnership contracts
            
            Simply add more PDFs to the project folder!
            """)
        
        # Feedback Section
        st.markdown("---")
        st.markdown("### 📝 Feedback")
        st.markdown("Help us improve LegalLens!")
        
        with st.expander("📋 Share Your Feedback", expanded=False):
            st.markdown("""
            We'd love to hear from you! Your feedback helps us make LegalLens better.
            """)
            # Embedded Google Form
            st.components.v1.iframe(
                "https://docs.google.com/forms/d/e/1FAIpQLSciPK7YKj9G9qAk8yDnLSBZ1Xvjwp36tecJOAIdVfvdwKnd4Q/viewform?embedded=true",
                height=600,
                scrolling=True
            )
        
        # Quick link button as alternative
        st.link_button(
            "🔗 Open in New Tab",
            "https://forms.gle/MN9ArBUhpC8T4qz69",
            use_container_width=True
        )

    # ===== BRANDED HEADER =====
    st.markdown('<h1 class="main-header">⚖️ LegalLens</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Multimodal Contract Analyzer • Understand contracts in seconds</p>', unsafe_allow_html=True)
    
    # Main Content - Welcome Screen when no contract
    if not contract_bytes:
        # Hero Section
        st.markdown("""
        <div class="hero-section">
            <h2>👋 Welcome to LegalLens</h2>
            <p style="color: #94a3b8; font-size: 1.1rem; margin: 1rem 0;">
                Your AI-powered legal assistant for contract analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.warning("👈 **Get Started:** Check the 'Use Demo Contract' box in the sidebar, or upload your own!")
        
        # Feature Cards
        st.markdown("### ✨ What You Can Do")
        
        feat_col1, feat_col2 = st.columns(2)
        
        with feat_col1:
            st.markdown("""
            <div class="feature-card">
                <h4>📄 Document Analysis</h4>
                <p style="color: #94a3b8;">View contracts with highlighted risk terms and get instant explanations</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>🎯 Risk Assessment</h4>
                <p style="color: #94a3b8;">AI-powered risk scoring with confidence levels and actionable insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
            <div class="feature-card">
                <h4>💬 Smart Chat</h4>
                <p style="color: #94a3b8;">Ask questions about your contract using natural language</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card">
                <h4>🎤 Voice Mode</h4>
                <p style="color: #94a3b8;">Speak your questions and get instant answers hands-free</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Start Guide
        with st.expander("📖 Quick Start Guide", expanded=False):
            st.markdown("""
            **Step 1:** Load a contract from the sidebar (demo or upload your own)
            
            **Step 2:** Start with the **Document Viewer** tab to see highlighted risk terms
            
            **Step 3:** Check the **Risk Analysis** tab for an AI-powered risk score
            
            **Step 4:** Use **Chat** or **Voice** to ask specific questions
            
            ---
            
            💡 **Pro Tip:** The Document Viewer automatically highlights 10+ legal risk terms like 
            "indemnity", "liability", "termination", and more!
            """)
        
        return
    
    # Process PDF
    file_hash = hashlib.sha256(contract_bytes).hexdigest()
    st.session_state["current_file_hash"] = file_hash

    try:
        vector_store, _, embedding_details = build_vector_store(file_hash, contract_bytes)
    except RuntimeError as err:
        st.error(f"❌ {err}")
        st.stop()

    _precompute_quick_actions(file_hash, vector_store)

    notice_key = "embedding_notice_shown"
    if embedding_details.get("embedding_source") != "openai":
        if not st.session_state.get(notice_key):
            _show_ephemeral_message(
                "OpenAI embeddings are unavailable. Using local SentenceTransformer embeddings instead.",
                level="warning"
            )
            if embedding_details.get("fallback_error"):
                _show_ephemeral_message(
                    f"OpenAI response: {embedding_details['fallback_error']}",
                    level="info"
                )
            st.session_state[notice_key] = True
    else:
        st.session_state[notice_key] = False
    
    # Document summary - Compact info bar
    st.markdown("---")
    doc_col1, doc_col2, doc_col3, doc_col4 = st.columns(4)
    with doc_col1:
        st.metric("📄 Document", contract_label or "Contract")
    with doc_col2:
        st.metric("📑 Pages", embedding_details.get("page_count", "-"))
    with doc_col3:
        size_kb = embedding_details.get("file_size_kb")
        st.metric("📊 Size", f"{size_kb} KB" if size_kb else "-")
    with doc_col4:
        st.metric("🧠 Embeddings", embedding_details.get("embedding_source", "-"))

    # ===== MAIN TABS - Reorganized for better UX flow =====
    # User Story: User uploads doc → Views it with highlights → Checks risk → Asks questions
    tab_doc, tab_risk, tab_chat, tab_voice = st.tabs([
        "📄 Document Viewer", 
        "🎯 Risk Analysis", 
        "💬 Chat", 
        "🎤 Voice"
    ])
    
    # ===== TAB 1: DOCUMENT VIEWER (PRIMARY - Most used) =====
    with tab_doc:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #3b82f6, #8b5cf6); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">🔦 Smart Document Viewer</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
                View your contract with automatically highlighted risk terms
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize highlight state
        if "highlight_generated" not in st.session_state:
            st.session_state.highlight_generated = False
            st.session_state.highlight_path = None
            st.session_state.highlight_hits = []
        
        # Auto-generate or button to generate
        col_action1, col_action2 = st.columns([2, 1])
        
        with col_action1:
            generate_highlight = st.button(
                "🔍 Generate Highlighted Document", 
                use_container_width=True, 
                type="primary",
                help="Highlights terms like indemnity, liability, termination, etc."
            )
        
        with col_action2:
            auto_highlight = st.checkbox("Auto-highlight on load", value=False)
        
        # Generate highlights
        should_generate = generate_highlight or (auto_highlight and not st.session_state.highlight_generated)
        
        if should_generate:
            with st.spinner("🎨 Analyzing and highlighting key legal terms..."):
                hits = _detect_highlight_terms(contract_bytes)
                tmp_pdf = Path(tempfile.gettempdir()) / f"contract_{file_hash}.pdf"
                tmp_pdf.write_bytes(contract_bytes)
                terms = [item["term"] for item in HIGHLIGHT_TERMS]
                annotated_path = highlight_terms_in_pdf(tmp_pdf, terms)
                
                st.session_state.highlight_generated = True
                st.session_state.highlight_path = annotated_path
                st.session_state.highlight_hits = hits
        
        # Display highlighted document
        if st.session_state.highlight_generated and st.session_state.highlight_path:
            st.success("✅ Document analyzed! Risk terms are highlighted in red.")
            
            # Show detected terms first (above PDF for visibility)
            hits = st.session_state.highlight_hits
            if hits:
                st.markdown("#### ⚠️ Risk Terms Found in Document")
                
                risk_cols = st.columns(min(len(hits), 3))
                for idx, item in enumerate(hits):
                    with risk_cols[idx % 3]:
                        st.markdown(f"""
                        <div class="feature-card" style="border-left: 4px solid #f59e0b;">
                            <strong style="color: #f59e0b;">{item['term'].title()}</strong>
                            <p style="color: #94a3b8; font-size: 0.85rem; margin: 0.5rem 0 0 0;">{item['why']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("✅ No tracked risk terms detected in this document.")
            
            st.markdown("---")
            
            # PDF Viewer
            st.markdown("#### 📖 Annotated Document Preview")
            _render_pdf_inline(Path(st.session_state.highlight_path), show_download_first=True)
        
        else:
            # Show info when not generated yet
            st.info("👆 Click **Generate Highlighted Document** to analyze and highlight risk terms in your contract.")
            
            # Show what terms will be highlighted
            with st.expander("ℹ️ Terms that will be highlighted", expanded=False):
                term_cols = st.columns(2)
                for idx, item in enumerate(HIGHLIGHT_TERMS):
                    with term_cols[idx % 2]:
                        st.markdown(f"• **{item['term'].title()}** - {item['why']}")
    
    # ===== TAB 2: RISK ANALYSIS =====
    with tab_risk:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #ef4444, #f59e0b); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">🎯 AI-Powered Risk Assessment</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
                Get an intelligent risk score and detailed analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Calculate Risk Score", use_container_width=True, type="primary"):
            with st.spinner("🤖 AI is analyzing contract risks... This may take a moment."):
                risk_data = calculate_risk_score(file_hash, vector_store)
                
                # Store in session for persistence
                st.session_state.risk_data = risk_data
        
        # Display risk results if available
        if "risk_data" in st.session_state:
            risk_data = st.session_state.risk_data
            score = risk_data.get("score", 50)
            level = risk_data.get("level", "Medium")
            
            # Color coding
            if score <= 30:
                color = "#22c55e"
                emoji = "✅"
                risk_class = "risk-low"
            elif score <= 60:
                color = "#f59e0b"
                emoji = "⚠️"
                risk_class = "risk-medium"
            else:
                color = "#ef4444"
                emoji = "🚨"
                risk_class = "risk-high"
            
            # Visual Risk Score Card
            st.markdown(f"""
            <div class="risk-score-container {risk_class}">
                <h1 style="font-size: 4rem; color: {color}; margin: 0;">{emoji} {score}</h1>
                <h2 style="color: {color}; margin: 0.5rem 0;">{level} Risk</h2>
                <p style="color: #94a3b8;">out of 100</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar visualization
            st.markdown("#### Risk Level Meter")
            st.progress(score / 100)
            
            # Confidence
            conf = risk_data.get("confidence")
            conf_level = risk_data.get("confidence_level")
            if conf is not None:
                conf_col1, conf_col2 = st.columns([1, 2])
                with conf_col1:
                    st.metric("Confidence", f"{conf:.0%}")
                with conf_col2:
                    st.caption(f"Confidence Level: {conf_level or 'Unknown'}")
                    st.progress(min(1.0, max(0.0, conf)))
            
            st.markdown("---")
            
            # Top Risks in cards
            st.markdown("#### 🔴 Top Identified Risks")
            
            risks_list = risk_data.get("top_risks", [])
            if risks_list:
                for i, risk in enumerate(risks_list, 1):
                    if isinstance(risk, dict):
                        issue = risk.get("issue") or "Unspecified risk"
                        evidence = risk.get("evidence")
                        severity = risk.get("severity")
                        recommendation = risk.get("recommendation")
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="feature-card" style="border-left: 4px solid {color};">
                                <strong style="color: {color};">Risk #{i}: {issue}</strong>
                                {f'<span style="float: right; background: {color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;">{severity}</span>' if severity else ''}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if evidence:
                                st.caption(f"📌 Evidence: _{evidence}_")
                            if recommendation:
                                st.info(f"💡 Recommendation: {recommendation}")
                    else:
                        st.warning(f"{i}. {risk}")
            else:
                st.success("No significant risks identified.")
            
            # Rationale
            if risk_data.get("rationale"):
                with st.expander("📝 Analysis Rationale", expanded=False):
                    st.markdown(risk_data["rationale"])
        
        else:
            # Show prompt to calculate
            st.markdown("""
            <div class="feature-card" style="text-align: center; padding: 2rem;">
                <h4>🎯 Ready to Analyze</h4>
                <p style="color: #94a3b8;">Click the button above to get an AI-powered risk assessment of your contract.</p>
                <p style="color: #64748b; font-size: 0.85rem;">The AI will analyze clauses related to liability, termination, indemnity, and more.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Actions Section
        st.markdown("### ⚡ Quick Insights")
        
        if "quick_action_output" not in st.session_state:
            st.session_state.quick_action_output = None
            st.session_state.quick_action_type = None
            st.session_state.quick_action_title = None

        precomputed_actions = st.session_state.get("quick_action_cache", {}).get("data", {})
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="quick-action-btn">
                <span style="font-size: 2rem;">📋</span>
                <p style="margin: 0.5rem 0 0 0; font-weight: 600;">Summarize</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Generate Summary", key="btn_summary", use_container_width=True):
                summary = precomputed_actions.get("summary")
                if not summary:
                    with st.spinner("Analyzing..."):
                        summary = get_rag_response(vector_store, QUICK_ACTION_PROMPTS["summary"])
                st.session_state.quick_action_output = summary
                st.session_state.quick_action_type = "info"
                st.session_state.quick_action_title = "📋 Contract Summary"

        with col2:
            st.markdown("""
            <div class="quick-action-btn">
                <span style="font-size: 2rem;">🚩</span>
                <p style="margin: 0.5rem 0 0 0; font-weight: 600;">Red Flags</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Find Red Flags", key="btn_risks", use_container_width=True):
                risks = precomputed_actions.get("risks")
                if not risks:
                    with st.spinner("Scanning for risks..."):
                        risks = get_rag_response(vector_store, QUICK_ACTION_PROMPTS["risks"])
                st.session_state.quick_action_output = risks
                st.session_state.quick_action_type = "warning"
                st.session_state.quick_action_title = "🚩 Red Flags & Concerns"

        with col3:
            st.markdown("""
            <div class="quick-action-btn">
                <span style="font-size: 2rem;">📅</span>
                <p style="margin: 0.5rem 0 0 0; font-weight: 600;">Key Dates</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Extract Dates", key="btn_dates", use_container_width=True):
                dates = precomputed_actions.get("dates")
                if not dates:
                    with st.spinner("Extracting dates..."):
                        dates = get_rag_response(vector_store, QUICK_ACTION_PROMPTS["dates"])
                st.session_state.quick_action_output = dates
                st.session_state.quick_action_type = "info"
                st.session_state.quick_action_title = "📅 Important Dates & Deadlines"

        # Display quick action result
        if st.session_state.quick_action_output:
            st.markdown("---")
            title = st.session_state.get("quick_action_title", "Details")
            level = st.session_state.get("quick_action_type", "info")
            result_text = st.session_state.quick_action_output
            
            st.markdown(f"#### {title}")
            if level == "warning":
                st.warning(result_text)
            else:
                st.info(result_text)
    
    # ===== TAB 3: CHAT =====
    with tab_chat:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #06b6d4, #3b82f6); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">💬 Chat with Your Contract</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
                Ask any question about your contract in natural language
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Suggested questions
        with st.expander("💡 Suggested Questions", expanded=False):
            st.markdown("""
            - What are the main obligations of each party?
            - What happens if the contract is terminated early?
            - Are there any automatic renewal clauses?
            - What are the payment terms?
            - What limitations of liability exist?
            - Is there a non-compete or exclusivity clause?
            """)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Clear chat button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History", use_container_width=False):
                st.session_state.messages = []
                st.rerun()
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about your contract..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing your contract..."):
                    response = get_rag_response(vector_store, question)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # ===== TAB 4: VOICE MODE =====
    with tab_voice:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #8b5cf6, #ec4899); padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0;">🎤 Voice Question & Answer</h3>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
                Speak your question and get an instant AI-powered answer
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div class="feature-card">
            <h4>📋 How to Use</h4>
            <ol style="color: #94a3b8; margin: 0.5rem 0;">
                <li>Click the microphone button below to start recording</li>
                <li>Ask your question clearly</li>
                <li>Click "Process Voice Question" to get your answer</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        if "voice_audio_bytes" not in st.session_state:
            st.session_state.voice_audio_bytes = None
            st.session_state.voice_result = None
        if "voice_clear_pending" not in st.session_state:
            st.session_state.voice_clear_pending = False

        if st.session_state.voice_clear_pending:
            st.session_state.pop("voice_recorder", None)
            st.session_state.voice_clear_pending = False

        # Audio recorder
        st.markdown("#### 🎙️ Record Your Question")
        audio_input = st.audio_input("Click to record", key="voice_recorder")

        if audio_input is not None:
            st.session_state.voice_audio_bytes = audio_input.read()
            st.session_state.voice_result = None

        if st.session_state.voice_audio_bytes:
            st.markdown("#### 🔊 Your Recording")
            st.audio(st.session_state.voice_audio_bytes)
            
            controls_col1, controls_col2 = st.columns(2)
            with controls_col1:
                if st.button("✨ Process Voice Question", use_container_width=True, type="primary"):
                    with st.spinner("🎧 Transcribing and analyzing..."):
                        result = get_audio_response(st.session_state.voice_audio_bytes, vector_store)
                        st.session_state.voice_result = result
            with controls_col2:
                if st.button("🗑️ Delete Recording", use_container_width=True):
                    st.session_state.voice_audio_bytes = None
                    st.session_state.voice_result = None
                    st.session_state.voice_clear_pending = True
                    st.rerun()

        # Display result
        result = st.session_state.get("voice_result")
        if result:
            st.markdown("---")
            if result.get("error"):
                st.error(f"❌ {result['error']}")
            else:
                if result.get("question"):
                    st.markdown("#### 🎙️ Transcribed Question")
                    st.info(result["question"])
                
                st.markdown("#### 📝 AI Answer")
                st.success(result.get("answer", ""))
        
        elif not st.session_state.voice_audio_bytes:
            st.markdown("""
            <div class="feature-card" style="text-align: center; padding: 2rem; border: 2px dashed #475569;">
                <span style="font-size: 3rem;">🎤</span>
                <p style="color: #94a3b8; margin: 1rem 0 0 0;">Click the microphone above to start recording your question</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
