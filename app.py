import streamlit as st
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pypdf import PdfReader
from supabase import create_client, Client


THEME_CSS = {
    "Dark": """
    <style>
        .stApp {
            background: linear-gradient(145deg, #0f172a 0%, #1f2937 45%, #111827 100%);
            color: #e2e8f0;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            color: #e2e8f0 !important;
            font-family: 'Segoe UI', sans-serif;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: rgba(148, 163, 184, 0.12);
            border-radius: 10px;
            color: #e2e8f0;
            padding: 12px 24px;
            border: 1px solid rgba(148, 163, 184, 0.15);
        }

        .stTabs [aria-selected="true"] {
            background-color: #2563eb;
            color: #f8fafc;
            border-color: rgba(37, 99, 235, 0.35);
        }

        footer {
            visibility: hidden;
        }

        [data-testid="stSidebar"] {
            background-color: #111827;
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }

        .css-1r6slb0, .stMarkdown, .stDataFrame, .stTable {
            background-color: rgba(17, 24, 39, 0.55);
            border-radius: 12px;
            padding: 18px;
            border: 1px solid rgba(148, 163, 184, 0.15);
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            color: #f8fafc;
            border: none;
            border-radius: 10px;
        }

        .stAlert {
            background-color: rgba(30, 64, 175, 0.35);
            border: 1px solid rgba(96, 165, 250, 0.4);
            border-radius: 10px;
        }
    </style>
    """,
}

QUICK_ACTION_PROMPTS = {
    "summary": "Provide a comprehensive summary of this contract, including parties involved, main obligations, and key terms.",
    "risks": "Identify any potential legal risks, red flags, or concerning clauses in this contract. Be specific and cite sections.",
    "dates": "List all important dates, deadlines, and time-sensitive obligations mentioned in this contract.",
}

LLM_MODEL = "gpt-4o-mini"
LLM_MAX_TOKENS = 450
RISK_MAX_TOKENS = 300

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


OPENAI_API_KEY = _fetch_openai_key()

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API Key not found. Please configure it in Streamlit secrets or environment variables.")
    st.stop()

# Ensure downstream libraries pick up the key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
    cache = st.session_state.get("quick_action_cache", {})
    if cache.get("hash") == file_hash and cache.get("data"):
        return

    results = {}

    def _run_query(prompt_key: str, prompt_text: str):
        try:
            return get_rag_response(vector_store, prompt_text)
        except Exception as err:
            return f"⚠️ Unable to generate {prompt_key.replace('_', ' ')}: {err}"

    with ThreadPoolExecutor(max_workers=len(QUICK_ACTION_PROMPTS)) as executor:
        futures = {
            key: executor.submit(_run_query, key, prompt)
            for key, prompt in QUICK_ACTION_PROMPTS.items()
        }
        for key, future in futures.items():
            results[key] = future.result()

    st.session_state["quick_action_cache"] = {
        "hash": file_hash,
        "data": results,
    }


def _render_auth_panel(supabase: Client) -> None:
    st.title("⚖️ LegalLens")
    st.markdown("### Secure Access")
    st.markdown("Sign in with your email or continue with Google.")

    tab_login, tab_signup, tab_google = st.tabs(["Sign In", "Create Account", "Google"])

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
                except Exception as err:
                    st.error(f"Sign-up failed: {err}")

    with tab_google:
        st.markdown("#### Continue with Google")
        redirect_to = _get_supabase_redirect_url()
        if st.button("Sign in with Google", use_container_width=True):
            try:
                response = supabase.auth.sign_in_with_oauth({
                    "provider": "google",
                    "options": {"redirect_to": redirect_to},
                })
                auth_url = _extract_attr(response, "url")
                if auth_url:
                    st.write("Redirecting to Google...")
                    st.markdown(f'<meta http-equiv="refresh" content="0; url={auth_url}" />', unsafe_allow_html=True)
                else:
                    st.warning("Received an unexpected response from Supabase. Please try again in a moment.")
            except Exception as err:
                st.error(f"Unable to start Google sign-in: {err}")

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
    ]

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    _trigger_rerun()

# ===========================
# TEXT RAG LOGIC
# ===========================
@st.cache_resource(show_spinner=False)
def _get_openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=OPENAI_API_KEY,
    )


@st.cache_resource(show_spinner=False)
def _get_local_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def build_vector_store(file_hash: str, file_bytes: bytes):
    """Extract text, split, and create FAISS vector store for a contract."""
    pdf_reader = PdfReader(BytesIO(file_bytes))
    text_chunks = []
    for page in pdf_reader.pages:
        text_chunks.append(page.extract_text())
    text = "".join(text_chunks)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    vector_store = None
    last_error = None
    embedding_source = "openai"

    try:
        openai_embeddings = _get_openai_embeddings()
        vector_store = FAISS.from_texts(chunks, openai_embeddings)
    except Exception as err:
        last_error = err

    if vector_store is None:
        embedding_source = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            local_embeddings = _get_local_embeddings()
            vector_store = FAISS.from_texts(chunks, local_embeddings)
        except Exception as local_err:
            raise RuntimeError(f"Fallback embeddings failed: {local_err}") from local_err

    embedding_details = {
        "embedding_source": embedding_source,
        "fallback_error": str(last_error) if last_error else None,
        "file_hash": file_hash,
    }

    return vector_store, text, embedding_details

def get_rag_response(vector_store, question):
    """Generate RAG-based response with legal contract analysis."""
    # Custom prompt template
    prompt_template = """You are LegalLens, an expert AI legal contract analyzer. 
    Use the following contract context to answer the question. 
    
    IMPORTANT INSTRUCTIONS:
    - Cite specific sections or clauses when possible
    - Warn about potential legal risks or red flags
    - If the answer isn't in the context, say so clearly
    - Be professional and precise
    
    Context: {context}
    
    Question: {question}
    
    Detailed Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.3,
        max_tokens=LLM_MAX_TOKENS,
        timeout=30,
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    response = qa_chain.invoke({"query": question})
    return response["result"]

def calculate_risk_score(vector_store):
    """Generate AI-powered contract risk score (0-100)."""
    risk_prompt = """Analyze this contract and provide a risk assessment score from 0-100 where:
    - 0-30: Low Risk (fair, balanced terms)
    - 31-60: Medium Risk (some concerning clauses)
    - 61-100: High Risk (unfavorable, dangerous terms)
    
    Evaluate based on:
    1. Unfair termination clauses
    2. Liability imbalances
    3. Payment terms
    4. Dispute resolution
    5. Hidden obligations
    6. Missing protections
    
    Return ONLY a JSON object with this exact format:
    {"score": <number>, "level": "<Low/Medium/High>", "top_risks": ["risk1", "risk2", "risk3"]}
    """
    
    llm = ChatOpenAI(
        model=LLM_MODEL,
        openai_api_key=OPENAI_API_KEY,
        temperature=0.2,
        max_tokens=RISK_MAX_TOKENS,
        timeout=30,
        response_format={"type": "json_object"},
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5})
    )
    
    try:
        response = qa_chain.invoke({"query": risk_prompt})
        result_text = response["result"]
        
        import json

        return json.loads(result_text)
    except Exception:
        return {"score": 50, "level": "Medium", "top_risks": ["Error analyzing contract"]}

# ===========================
# AUDIO LOGIC
# ===========================
def get_audio_response(audio_bytes, vector_store):
    """Transcribe audio with OpenAI and answer via RAG."""
    audio_buffer = BytesIO(audio_bytes)
    audio_buffer.name = "question.wav"

    try:
        transcription = openai_client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_buffer
        )
        question_text = transcription.text.strip()
    except Exception as err:
        return {"error": f"❌ Error transcribing audio: {err}", "question": None}

    if not question_text:
        return {"error": "⚠️ Unable to understand the audio. Please try speaking clearly again.", "question": None}

    try:
        answer = get_rag_response(vector_store, question_text)
        return {"question": question_text, "answer": answer}
    except Exception as err:
        return {"error": f"❌ Error generating answer: {err}", "question": question_text}

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

    with st.sidebar:
        st.markdown(f"👤 **Signed in as:** `{username}`")
        if st.button("🔓 Log Out", use_container_width=True):
            _logout(supabase)
            st.stop()

        st.markdown("---")
        st.header("📄 Contract Database")
        
        # Option to use default or upload
        use_default = st.checkbox("📚 Use Demo Contract (act.pdf)", value=True)
        
        if use_default:
            if os.path.exists("act.pdf"):
                with open("act.pdf", "rb") as f:
                    contract_bytes = f.read()
                st.success("✅ Loaded: act.pdf (Demo)")
                st.info("💡 Uncheck to upload your own contract")
            else:
                st.error("❌ Demo file 'act.pdf' not found!")
        else:
            uploaded_file = st.file_uploader(
                "Upload a PDF contract",
                type=["pdf"],
                help="Upload a legal contract in PDF format"
            )
            
            if uploaded_file:
                contract_bytes = uploaded_file.getvalue()
                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.info(f"📊 Size: {uploaded_file.size / 1024:.2f} KB")
        
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

    # Header
    st.title("⚖️ LegalLens")
    st.markdown("### AI-Powered Multimodal Contract Analyzer")
    st.markdown("---")
    
    # Main Content
    if not contract_bytes:
        st.warning("👈 Check the 'Use Demo Contract' box in the sidebar to begin, or upload your own!")
        st.markdown("""
        ### Welcome to LegalLens! 🚀
        
        **Features:**
        - 💬 **Text Chat**: Ask questions about your contract using RAG technology
        - 🎤 **Voice Mode**: Speak your questions naturally
        - 🔍 **Smart Analysis**: AI-powered insights with section citations
        - ⚠️ **Risk Detection**: Automatic red flag identification
        
        **📚 Demo Database**: We've pre-loaded `act.pdf` for instant testing!
        """)
        return
    
    # Process PDF
    file_hash = hashlib.sha256(contract_bytes).hexdigest()

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
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 Chat", "🎤 Voice Mode"])
    
    # TAB 1: DASHBOARD
    with tab1:
        st.header("Quick Contract Analysis")
        
        # AI Risk Score Section
        st.markdown("### 🎯 AI-Powered Risk Assessment")
        
        if st.button("🔍 Calculate Risk Score", use_container_width=True, type="primary"):
            with st.spinner("🤖 AI is analyzing contract risks..."):
                risk_data = calculate_risk_score(vector_store)
                
                # Display risk score with visual meter
                score = risk_data.get("score", 50)
                level = risk_data.get("level", "Medium")
                
                # Color coding
                if score <= 30:
                    color = "#28a745"  # Green
                    emoji = "✅"
                elif score <= 60:
                    color = "#ffc107"  # Yellow
                    emoji = "⚠️"
                else:
                    color = "#dc3545"  # Red
                    emoji = "🚨"
                
                # Visual Risk Meter
                col_score, col_gauge = st.columns([1, 2])
                
                with col_score:
                    st.markdown(f"<h1 style='text-align: center; color: {color};'>{emoji} {score}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center; color: {color};'>{level} Risk</h3>", unsafe_allow_html=True)
                
                with col_gauge:
                    # Progress bar as risk meter
                    st.markdown("#### Risk Level Meter")
                    st.progress(score / 100)
                    st.caption(f"Score: {score}/100")
                
                # Top Risks
                st.markdown("#### 🔴 Top Identified Risks:")
                for i, risk in enumerate(risk_data.get("top_risks", []), 1):
                    st.warning(f"{i}. {risk}")
        
        st.markdown("---")

        # Quick Action Buttons
        st.markdown("### ⚡ Quick Actions")
        if "quick_action_output" not in st.session_state:
            st.session_state.quick_action_output = None
            st.session_state.quick_action_type = None
            st.session_state.quick_action_title = None

        precomputed_actions = st.session_state.get("quick_action_cache", {}).get("data", {})
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔍 Summarize Contract", use_container_width=True):
                summary = precomputed_actions.get("summary")
                if not summary:
                    with st.spinner("Analyzing..."):
                        summary = get_rag_response(vector_store, QUICK_ACTION_PROMPTS["summary"])
                st.session_state.quick_action_output = summary
                st.session_state.quick_action_type = "info"
                st.session_state.quick_action_title = "Summary"

        with col2:
            if st.button("⚠️ Identify Red Flags", use_container_width=True):
                risks = precomputed_actions.get("risks")
                if not risks:
                    with st.spinner("Scanning for risks..."):
                        risks = get_rag_response(vector_store, QUICK_ACTION_PROMPTS["risks"])
                st.session_state.quick_action_output = risks
                st.session_state.quick_action_type = "warning"
                st.session_state.quick_action_title = "Risk Analysis"

        with col3:
            if st.button("📅 Extract Key Dates", use_container_width=True):
                dates = precomputed_actions.get("dates")
                if not dates:
                    with st.spinner("Extracting dates..."):
                        dates = get_rag_response(vector_store, QUICK_ACTION_PROMPTS["dates"])
                st.session_state.quick_action_output = dates
                st.session_state.quick_action_type = "info"
                st.session_state.quick_action_title = "Important Dates"

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
    
    # TAB 2: CHAT
    with tab2:
        st.header("Chat with Your Contract")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
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
                with st.spinner("🤔 Analyzing..."):
                    response = get_rag_response(vector_store, question)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # TAB 3: VOICE MODE
    with tab3:
        st.header("Voice Question & Answer")
        st.markdown("🎤 **Speak your question** and get an instant answer!")
        
        if "voice_audio_bytes" not in st.session_state:
            st.session_state.voice_audio_bytes = None
            st.session_state.voice_result = None
        if "voice_clear_pending" not in st.session_state:
            st.session_state.voice_clear_pending = False

        if st.session_state.voice_clear_pending:
            st.session_state.pop("voice_recorder", None)
            st.session_state.voice_clear_pending = False

        audio_input = st.audio_input("Record your question", key="voice_recorder")

        if audio_input is not None:
            st.session_state.voice_audio_bytes = audio_input.read()
            st.session_state.voice_result = None

        if st.session_state.voice_audio_bytes:
            st.audio(st.session_state.voice_audio_bytes)
            controls_col1, controls_col2 = st.columns(2)
            with controls_col1:
                if st.button("🔊 Process Voice Question", use_container_width=True):
                    with st.spinner("🎧 Processing audio..."):
                        result = get_audio_response(st.session_state.voice_audio_bytes, vector_store)
                        st.session_state.voice_result = result
            with controls_col2:
                if st.button("🗑️ Delete Recording", use_container_width=True):
                    st.session_state.voice_audio_bytes = None
                    st.session_state.voice_result = None
                    st.session_state.voice_clear_pending = True

        result = st.session_state.get("voice_result")
        if result:
            if result.get("error"):
                st.error(result["error"])
            else:
                if result.get("question"):
                    st.markdown("#### 🎙️ Transcribed Question")
                    st.info(result["question"])

                st.markdown("### 📝 Answer:")
                st.success(result.get("answer", ""))

if __name__ == "__main__":
    main()
