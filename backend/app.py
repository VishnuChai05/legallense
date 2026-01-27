import hashlib
import os
import time
import json
from functools import lru_cache
from io import BytesIO
from typing import Dict, Tuple

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import redis
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()

app = FastAPI(title="LegalLens API", version="0.1.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    return response

# ===========================
# CONFIG
# ===========================
LLM_MODEL_PRIMARY = "gpt-4o-mini"
LLM_MODEL_FALLBACK = "gpt-3.5-turbo"
LLM_MAX_TOKENS = 500
# Lowered to keep responses small for parsing
RISK_MAX_TOKENS = 150

# Optional local model for risk route (e.g., Ollama). Leave empty to skip.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")  # e.g., "llama3.1:8b"; empty disables

EMBEDDING_MODEL_PRIMARY = "text-embedding-3-small"
EMBEDDING_MODEL_FALLBACK = "text-embedding-3-small"

# Risk retrieval focus topics to widen coverage
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

# Confidence mapping helper
def _confidence_level(score: float) -> str:
    if score >= 0.67:
        return "High"
    if score >= 0.34:
        return "Medium"
    return "Low"

MAX_FILE_SIZE_MB = 10
MAX_PDF_PAGES = 200
MAX_QUESTION_CHARS = 800
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "20"))
CACHE_TTL_SECONDS = 900
CACHE_MAX_ITEMS = 8
REDIS_URL = os.getenv("REDIS_URL", "").strip()


redis_client = None
fallback_rate_state: Dict[str, list[float]] = {}


def _get_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    return key


def _get_redis():
    global redis_client
    if not REDIS_URL:
        return None
    if redis_client is None:
        redis_client = redis.Redis.from_url(REDIS_URL)
    return redis_client

# ===========================
# STORAGE (in-memory for now)
# ===========================
vector_store_cache: Dict[str, tuple[FAISS, float]] = {}
contract_bytes_cache: Dict[str, tuple[bytes, float]] = {}
rate_limit_state: Dict[str, list[float]] = {}


def _use_ollama() -> bool:
    return bool(OLLAMA_MODEL.strip())


def _ollama_generate(prompt: str, max_tokens: int) -> str:
    """Call local Ollama model; raises on error."""

    if not _use_ollama():
        raise RuntimeError("OLLAMA_MODEL not configured")

    url = f"{OLLAMA_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL.strip(),
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
        },
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("response", "")
        if not text:
            raise RuntimeError("Empty response from Ollama")
        return text

# ===========================
# HELPERS
# ===========================

def _validate_pdf_upload(uploaded_file: UploadFile) -> bytes:
    content_type = (uploaded_file.content_type or "").lower()
    filename = (uploaded_file.filename or "").lower()

    if content_type not in ("application/pdf", "application/octet-stream"):
        if not filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if content_type == "application/octet-stream" and not filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = uploaded_file.file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_FILE_SIZE_MB} MB")
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if not contents.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Invalid PDF file")

    try:
        pdf_reader = PdfReader(BytesIO(contents))
        if len(pdf_reader.pages) > MAX_PDF_PAGES:
            raise HTTPException(status_code=400, detail=f"File too large. Max {MAX_PDF_PAGES} pages")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to read PDF")

    return contents


@lru_cache(maxsize=1)
def _get_openai_embeddings_small() -> OpenAIEmbeddings:
    key = _get_api_key()
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_PRIMARY,
        openai_api_key=key,
        chunk_size=500,
        request_timeout=60,
    )


@lru_cache(maxsize=1)
def _get_openai_embeddings_large() -> OpenAIEmbeddings:
    key = _get_api_key()
    return OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=key,
        chunk_size=500,
        request_timeout=60,
    )


@lru_cache(maxsize=1)
def _get_local_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def _evict_stale_cache():
    now = time.time()
    expired_keys = []
    for key, (_, ts) in vector_store_cache.items():
        if now - ts > CACHE_TTL_SECONDS:
            expired_keys.append(key)
    for key in expired_keys:
        vector_store_cache.pop(key, None)
        contract_bytes_cache.pop(key, None)

    # Enforce size cap (drop oldest first)
    if len(vector_store_cache) > CACHE_MAX_ITEMS:
        sorted_items = sorted(vector_store_cache.items(), key=lambda kv: kv[1][1])
        for key, _ in sorted_items[: len(vector_store_cache) - CACHE_MAX_ITEMS]:
            vector_store_cache.pop(key, None)
            contract_bytes_cache.pop(key, None)


def _cache_key(user_id: str, file_hash: str) -> str:
    return f"{user_id}:{file_hash}"


def _check_rate_limit(user_id: str, client_ip: str):
    redis_conn = _get_redis()
    key = f"rl:{user_id}:{client_ip}"
    limit = RATE_LIMIT_PER_MIN + RATE_LIMIT_BURST
    if redis_conn:
        count = redis_conn.incr(key)
        if count == 1:
            redis_conn.expire(key, 60)
        if count > limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
    else:
        now = time.time()
        window_start = now - 60
        entries = fallback_rate_state.get(key, [])
        entries = [ts for ts in entries if ts >= window_start]
        if len(entries) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        entries.append(now)
        fallback_rate_state[key] = entries


def _get_chat_llm() -> ChatOpenAI:
    try:
        key = _get_api_key()
        return ChatOpenAI(
            model=LLM_MODEL_PRIMARY,
            openai_api_key=key,
            temperature=0.3,
            max_tokens=LLM_MAX_TOKENS,
            timeout=30,
            request_timeout=30,
        )
    except Exception:
        key = _get_api_key()
        return ChatOpenAI(
            model=LLM_MODEL_FALLBACK,
            openai_api_key=key,
            temperature=0.3,
            max_tokens=LLM_MAX_TOKENS,
            timeout=20,
            request_timeout=20,
        )


def _get_risk_llm() -> ChatOpenAI:
    try:
        key = _get_api_key()
        return ChatOpenAI(
            model=LLM_MODEL_PRIMARY,
            openai_api_key=key,
            temperature=0.2,
            max_tokens=RISK_MAX_TOKENS,
            timeout=30,
            request_timeout=30,
            response_format={"type": "json_object"},
        )
    except Exception:
        key = _get_api_key()
        return ChatOpenAI(
            model=LLM_MODEL_FALLBACK,
            openai_api_key=key,
            temperature=0.2,
            max_tokens=RISK_MAX_TOKENS,
            timeout=20,
            request_timeout=20,
            response_format={"type": "json_object"},
        )


def _build_vector_store(user_id: str, file_hash: str, file_bytes: bytes):
    pdf_reader = PdfReader(BytesIO(file_bytes))
    text_chunks = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
    text = "\n".join(text_chunks)

    # Simple splitter (LangChain default split by RecursiveCharacterTextSplitter requires extra import);
    # keep minimal to avoid overloading local hardware.
    chunk_size = 1200
    overlap = 150
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    embedding_source = None
    errors = []
    embedding_attempts = [
        ("all-MiniLM-L6-v2 (local)", _get_local_embeddings),
        ("text-embedding-3-small", _get_openai_embeddings_small),
        ("text-embedding-3-large", _get_openai_embeddings_large),
    ]

    vector_store = None
    for source_name, get_embeddings_fn in embedding_attempts:
        try:
            embeddings = get_embeddings_fn()
            vector_store = FAISS.from_texts(chunks, embeddings)
            embedding_source = source_name
            break
        except Exception as err:
            errors.append(f"{source_name}: {err}")
            continue

    if vector_store is None:
        raise RuntimeError(f"All embeddings failed: {'; '.join(errors)}")

    now = time.time()
    cache_key = _cache_key(user_id, file_hash)
    vector_store_cache[cache_key] = (vector_store, now)
    contract_bytes_cache[cache_key] = (file_bytes, now)
    return {
        "embedding_source": embedding_source,
        "chunk_count": len(chunks),
        "file_hash": file_hash,
        "fallback_errors": errors[:-1] if len(errors) > 1 else None,
    }


def _build_rag_prompt(vector_store, question: str) -> str:
    docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)

    return f"""You are LegalLens, an expert AI legal contract analyzer with deep expertise in contract law.

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


def _execute_rag_query(vector_store, question: str) -> str:
    """Manual retrieve-then-generate RAG to reduce dependencies."""

    prompt = _build_rag_prompt(vector_store, question)
    llm = _get_chat_llm()
    result = llm.invoke(prompt)
    return getattr(result, "content", str(result))


def _stream_rag_query(vector_store, question: str):
    prompt = _build_rag_prompt(vector_store, question)
    llm = _get_chat_llm()
    try:
        for chunk in llm.stream(prompt):
            content = getattr(chunk, "content", None)
            if content:
                yield content
    except Exception:
        # Fallback to single response
        yield _execute_rag_query(vector_store, question)


def _collect_risk_context(vector_store, max_chars: int = 1200, max_snippets: int = 10) -> tuple[str, int]:
    """Gather risk-focused excerpts, but cap total size to avoid LLM length errors."""

    snippets = []
    seen = set()
    total_chars = 0

    for topic in RISK_TOPICS:
        for doc in vector_store.similarity_search(topic, k=2):
            key = hash(doc.page_content)
            if key in seen or len(snippets) >= max_snippets:
                continue
            text = doc.page_content or ""
            if not text:
                continue
            # Truncate individual snippets if they're very long
            text = text[: max_chars // 4]
            add_len = len(text) + 4  # include separator margin
            if total_chars + add_len > max_chars:
                continue
            seen.add(key)
            snippets.append(text)
            total_chars += add_len

    if not snippets:
        return "", 0
    return "\n\n---\n\n".join(snippets), len(snippets)


def _normalize_risk_payload(raw_text: str, snippet_count: int) -> dict:
    """Parse LLM JSON with guards and consistent shape."""

    def _level_from_score(score_val: int) -> str:
        if score_val <= 30:
            return "Low"
        if score_val <= 60:
            return "Medium"
        return "High"

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        data = json.loads(match.group()) if match else {}
    except Exception:
        data = {}

    score = data.get("score")
    try:
        score = int(round(float(score)))
    except Exception:
        score = 50
    score = max(0, min(100, score))

    level = data.get("level") or _level_from_score(score)
    top_risks = data.get("top_risks") or data.get("risks") or []
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

    # Derive confidence if model did not provide one
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


def _execute_risk_calculation(vector_store) -> dict:
    """Risk calculation via focused retrieval + structured prompt."""

    context, snippet_count = _collect_risk_context(vector_store)

    # Prefer local model (Ollama) for risk if configured
    if _use_ollama():
        try:
            prompt = f"""You are a contracts lawyer. Return strict JSON only.

CONTEXT:
{context}

JSON SCHEMA:
{{"score":0-100, "level":"Low|Medium|High", "rationale":"...", "top_risks":[{{"issue":"...","evidence":"..."}}], "confidence":0-1}}
Keep the response under 120 tokens and do not include anything outside JSON.
"""
            result_text = _ollama_generate(prompt, max_tokens=160)
            return _normalize_risk_payload(result_text, snippet_count)
        except Exception:
            # Fall through to remote model
            pass

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

Keep the JSON very short (<= 120 tokens total). Do not add extra text outside JSON.

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
    except Exception:
        # Retry with a much smaller context and prompt to avoid length errors
        small_context, small_count = _collect_risk_context(vector_store, max_chars=600, max_snippets=6)
        small_prompt = f"""You are a concise contracts lawyer. Return terse JSON only.

CONTEXT:
{small_context}

JSON SCHEMA:
{{"score": <0-100>, "level": "Low|Medium|High", "top_risks": [{{"issue": "...", "evidence": "..."}}], "confidence": <0-1>}}
Keep under 80 tokens total.
"""
        try:
            result = llm.invoke(small_prompt)
            result_text = getattr(result, "content", str(result))
            return _normalize_risk_payload(result_text, small_count)
        except Exception:
            # Final fallback: use a plain ChatOpenAI without JSON mode and with tiny max tokens
            try:
                key = _get_api_key()
                tiny_llm = ChatOpenAI(
                    model=LLM_MODEL_FALLBACK,
                    openai_api_key=key,
                    temperature=0.2,
                    max_tokens=80,
                    timeout=15,
                    request_timeout=15,
                    response_format=None,
                )
                tiny_prompt = f"Return JSON only: {{\"score\":0-100, \"level\":\"Low|Medium|High\", \"top_risks\":[{{\"issue\":\"...\",\"evidence\":\"...\"}}], \"confidence\":0-1}}. Context: {small_context[:400]}"
                result = tiny_llm.invoke(tiny_prompt)
                result_text = getattr(result, "content", str(result))
                return _normalize_risk_payload(result_text, small_count)
            except Exception as e:
                return {
                    "score": 50,
                    "level": "Medium",
                    "top_risks": [f"Error generating risk analysis: {str(e)[:80]}"],
                    "rationale": None,
                    "confidence": 0.2,
                    "confidence_level": "Low",
                }


# ===========================
# REQUEST MODELS
# ===========================
class ChatRequest(BaseModel):
    file_hash: str = Field(..., description="Hash of the uploaded file")
    question: str = Field(..., description="User question")


class RiskRequest(BaseModel):
    file_hash: str = Field(..., description="Hash of the uploaded file")


# ===========================
# ROUTES
# ===========================
def _get_vector_store(user_id: str, file_hash: str):
    _evict_stale_cache()
    key = _cache_key(user_id, file_hash)
    entry = vector_store_cache.get(key)
    if not entry:
        return None
    store, ts = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        vector_store_cache.pop(key, None)
        contract_bytes_cache.pop(key, None)
        return None
    return store


def _client_id_from_request(request: Request) -> str:
    client = request.client.host if request.client else "unknown"
    return client


def _user_id_from_request(request: Request) -> str:
    header_val = request.headers.get("x-user-id") if request else None
    if header_val:
        return header_val.strip() or "anon"
    return "anon"


@app.get("/health")
def health(request: Request):
    _check_rate_limit(_user_id_from_request(request), _client_id_from_request(request))
    return {"status": "ok"}


@app.post("/contracts/upload")
def upload_contract(request: Request, file: UploadFile = File(...)):
    user_id = _user_id_from_request(request)
    _check_rate_limit(user_id, _client_id_from_request(request))
    contents = _validate_pdf_upload(file)
    file_hash = hashlib.sha256(contents).hexdigest()
    meta = _build_vector_store(user_id, file_hash, contents)
    return {"file_hash": file_hash, "meta": meta}


@app.post("/chat")
def chat(request: Request, payload: ChatRequest):
    user_id = _user_id_from_request(request)
    _check_rate_limit(user_id, _client_id_from_request(request))
    if len(payload.question or "") > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Question too long")

    vector_store = _get_vector_store(user_id, payload.file_hash)
    if not vector_store:
        raise HTTPException(status_code=404, detail="Vector store not found; upload first")

    answer = _execute_rag_query(vector_store, payload.question)
    return {"answer": answer}


@app.post("/chat/stream")
def chat_stream(request: Request, payload: ChatRequest):
    user_id = _user_id_from_request(request)
    _check_rate_limit(user_id, _client_id_from_request(request))
    if len(payload.question or "") > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail="Question too long")

    vector_store = _get_vector_store(user_id, payload.file_hash)
    if not vector_store:
        raise HTTPException(status_code=404, detail="Vector store not found; upload first")

    def iter_tokens():
        for token in _stream_rag_query(vector_store, payload.question):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(iter_tokens(), media_type="text/event-stream")


@app.post("/risk")
def risk(request: Request, payload: RiskRequest):
    user_id = _user_id_from_request(request)
    _check_rate_limit(user_id, _client_id_from_request(request))
    vector_store = _get_vector_store(user_id, payload.file_hash)
    if not vector_store:
        raise HTTPException(status_code=404, detail="Vector store not found; upload first")

    result = _execute_risk_calculation(vector_store)
    return JSONResponse(result)
