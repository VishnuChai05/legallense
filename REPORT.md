# LegalLens: AI-Powered Multimodal Contract Analyzer

## Executive Summary
LegalLens is a production-ready Streamlit application designed to accelerate contract review using Retrieval-Augmented Generation (RAG), Supabase authentication, and voice interaction. It ingests PDF contracts, builds a FAISS vector index with OpenAI or local embeddings, and delivers grounded answers via GPT-4o mini with explicit citations. A quick-actions panel precomputes summaries, red flags, and key dates to reduce cognitive load and time-to-insight. The app is engineered for cloud deployment on Streamlit Community Cloud with robust secrets management, pinned runtime (Python 3.11.9), and dependency versions compatible with legacy LangChain APIs.

## 1. Introduction
- **Background**: Legal practitioners spend considerable effort scanning long contracts for obligations, risk clauses, and deadlines. Conventional manual review is slow, costly, and error-prone.
- **Motivation**: Modern LLMs can synthesize context effectively, but answers must be grounded in the source to be trusted. RAG provides a principled approach to bind answers to the contract text.
- **Project Vision**: Deliver an accessible, secure, and performant tool that allows legal teams to upload contracts, query them conversationally (text/voice), and obtain structured analyses (summaries, risks, dates) with citations.
- **Scope**: PDF ingestion, chunking, embedding retrieval, RAG question answering, voice transcription, Supabase Auth, caching, and deployability.
- **Assumptions**: Input contracts are machine-readable PDFs; users have OpenAI and Supabase credentials; network connectivity is available for cloud features.

## 2. Objectives & Success Criteria
- **Grounded Answers**: Provide responses that cite specific contract sections.
- **Usability**: Achieve smooth UX via quick actions and voice input.
- **Security**: Gate access with Supabase Auth; manage secrets securely.
- **Performance**: Minimize latency through caching and precomputation.
- **Deployability**: Reliable builds on Streamlit Cloud with pinned Python/runtime and compatible dependency versions.
- **Maintainability**: Clear documentation, modular code, and defensive error handling.

## 3. System Architecture Overview
- **Frontend**: Streamlit UI in `app.py` with a dark legal-tech theme.
- **Core Components**:
  - PDF ingestion: `pypdf.PdfReader` extracts text.
  - Text splitting: `langchain.text_splitter.RecursiveCharacterTextSplitter` balances chunk granularity and context overlap.
  - Embeddings: Primary `OpenAIEmbeddings` (Text Embedding 3 Large); fallback `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`).
  - Vector store: `langchain_community.vectorstores.FAISS` (CPU).
  - RAG QA: `langchain.chains.RetrievalQA` with `PromptTemplate` to enforce citations and cautious language.
  - LLM: `langchain_openai.ChatOpenAI` targeting `gpt-4o-mini` with token limits tailored for responses and risk outputs.
  - Voice: Audio capture in Streamlit; transcription with GPT-4o mini.
  - Auth: Supabase email/password & Google OAuth; session set via `exchange_code_for_session` and cached in `st.session_state`.
- **Caching Strategy**: `st.cache_resource` for embeddings/models; session-state caches for quick actions and auth tokens.
- **Configuration**: `.env` locally via `dotenv` and Streamlit `st.secrets` in cloud; redirect URL normalization and environment fallback logic.

## 4. Data Model & Processing Pipeline
- **Input**: PDF file uploaded via Streamlit sidebar.
- **Preprocessing**:
  - Extract text per page; normalize whitespace and punctuation as needed.
  - Compute file hash (e.g., SHA-256) to key caches across sessions.
- **Chunking**:
  - Parameters like `chunk_size` (~1000-1500 chars) and `chunk_overlap` (~100-200 chars) tuned to keep clauses together while enabling effective retrieval.
  - Recursive splitter preserves semantic boundaries better than naive fixed-size splits.
- **Embedding**:
  - For each chunk, compute embeddings using OpenAI; if exceptions occur (quota, network), fallback to local model.
  - Store embeddings in FAISS index (in-memory for current session).
- **Retrieval**:
  - Use FAISS to perform similarity search (top-k, typically k=4-8) to surface context relevant to the query.
- **Answer Synthesis**:
  - Construct a prompt with retrieved contexts, instructing the LLM to cite sections explicitly and avoid unsupported claims.
  - Generate concise, legally cautious responses with `LLM_MAX_TOKENS`; use `RISK_MAX_TOKENS` for compact risk items.

## 5. RAG Prompting & Citation Strategy
- **PromptTemplate Design**:
  - Include guidelines: “Ground answers in the provided excerpts,” “Quote or reference section numbers when available,” and “Flag uncertainties.”
  - Balance verbosity and precision to keep outputs readable.
- **Citation Rendering**:
  - Map answer snippets back to source chunk metadata (page/section).
  - Present citations inline or as a list beneath the answer card for clarity.
- **Failure Modes**:
  - Insufficient context: Prompt the user to upload a contract or ask a more specific question.
  - Ambiguity: Encourage follow-up questions; display uncertainty notes.

## 6. Voice Interaction Subsystem
- **Audio Capture**: Streamlit UI records short queries; bytes stored in `st.session_state`.
- **Transcription**: GPT-4o mini STT transforms audio to text; language auto-detection supported.
- **Answer Flow**: Transcription fed into the same RAG pipeline; results returned in text; optional TTS (future work) could synthesize spoken answers.
- **Resilience**: Timeout handling; clear error messages for mic permission or empty audio.

## 7. Authentication & Session Management
- **Supabase Email/Password**:
  - Sign-up: `supabase.auth.sign_up` with helpful feedback and email verification messaging.
  - Sign-in: `supabase.auth.sign_in_with_password` with errors surfaced to the user.
- **Google OAuth**:
  - Initiation: `supabase.auth.sign_in_with_oauth({ provider: 'google', options: { redirect_to } })`.
  - Callback: Parse `code` from Streamlit `st.query_params`, then `exchange_code_for_session` and set tokens.
- **Session State**:
  - Store `access_token`, `refresh_token`, and `username` in `st.session_state`.
  - Sync tokens with `supabase.auth.set_session` at start; clear on logout.
- **Redirect URL Management**:
  - Read from `st.secrets` first, fallback to `.env`; default to `http://localhost:8501` for development.
  - Strip trailing slashes to avoid mismatch.

## 8. Performance Engineering & Caching
- **Embeddings Cache**:
  - Use `st.cache_resource` to memoize models and vector stores keyed by document hash; cache invalidation occurs on new upload.
- **Quick Actions Precompute**:
  - Run summary, risk, and dates in parallel via `ThreadPoolExecutor` immediately after indexing finishes.
  - Cache results in `st.session_state['quick_action_cache']` with `{ hash, data }` structure.
- **Latency Considerations**:
  - Avoid recomputing embeddings for unchanged files.
  - Keep top-k small but sufficient; tune by empirical tests.
- **Resource Constraints**:
  - FAISS index stored in memory; suitable for single-document session use. For multi-document or persistence, consider on-disk serialization or a service.

## 9. Deployment & Operations
- **Cloud Platform**: Streamlit Community Cloud.
- **Runtime Pin**: `runtime.txt` → `python-3.11.9` to avoid Python 3.13 incompatibilities (as observed in logs).
- **Dependencies**: `requirements.txt` with compatible pins:
  - `langchain==0.1.20`, `langchain-openai==0.1.7`, `langchain-community==0.0.38`
  - `streamlit`, `pypdf`, `faiss-cpu`, `openai`, `python-dotenv`, `sentence-transformers`, `supabase`
- **Secrets (TOML)**:
  ```toml
  OPENAI_API_KEY = "sk-..."
  SUPABASE_URL = "https://<your-project>.supabase.co"
  SUPABASE_ANON_KEY = "<your-anon-key>"
  SUPABASE_REDIRECT_URL = "https://<your-app>.streamlit.app"
  ```
  - Quotes required; redirect must match Supabase settings.
- **Git Workflow**: `git add`, `git commit`, `git push` on branch `master`, then trigger Redeploy.
- **Troubleshooting Playbook**:
  - Dependency conflicts: Adjust pins to satisfy solver (e.g., `langchain-community>=0.0.38`).
  - Missing modules: Ensure `langchain` core and related packages present.
  - Runtime mismatch: Confirm `runtime.txt` is present and valid.
  - OAuth issues: Verify exact redirect URL in both Streamlit secrets and Supabase dashboard.

### 9.1 Operations Deep Dive
- **Git Commands (Windows PowerShell)**:
  ```powershell
  git add .
  git commit -m "Deploy update"
  git push
  ```
  Use `;` to chain in PowerShell if needed, not `&&`.

- **Redeploy Steps (Streamlit Cloud)**:
  1. Open your app dashboard on Streamlit Cloud.
  2. Click the overflow menu → Redeploy/Reboot to trigger rebuild.
  3. Watch logs for dependency resolution and runtime selection.

- **Secrets Configuration (TOML)**:
  - Ensure all values are quoted, for example:
    ```toml
    OPENAI_API_KEY = "sk-..."
    SUPABASE_URL = "https://<project>.supabase.co"
    SUPABASE_ANON_KEY = "..."
    SUPABASE_REDIRECT_URL = "https://<app>.streamlit.app"
    ```
  - Redirect URL must match Supabase Auth settings exactly.

- **Typical Cloud Log Issues & Fixes**:
  - "No solution found when resolving dependencies": Pins conflict → align versions (e.g., `langchain==0.1.20` with `langchain-community>=0.0.38`).
  - "ModuleNotFoundError: langchain.text_splitter": Newer LangChain removed legacy paths → pin to `0.1.x`.
  - Python wheels missing (e.g., `faiss-cpu`): Use Python 3.11 runtime via `runtime.txt`.
  - OAuth code not exchanged: Confirm `SUPABASE_REDIRECT_URL` in both secrets and Supabase dashboard.

- **Local Run & Verification**:
  ```powershell
  # Create venv and install deps
  python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt

  # Set local env
  $env:OPENAI_API_KEY = "sk-..."; $env:SUPABASE_URL = "https://<project>.supabase.co"; $env:SUPABASE_ANON_KEY = "..."; $env:SUPABASE_REDIRECT_URL = "http://localhost:8501"

  # Run the app
  streamlit run app.py
  ```

- **Version Audit**:
  ```powershell
  pip freeze | Select-String "langchain|faiss|streamlit|openai|sentence-transformers"
  ```

- **Rollback Strategy**:
  - If a deploy breaks due to pins, revert to last known good commit in Git.
  - Keep `requirements.txt` and `runtime.txt` stable; change one variable at a time.

## 10. Security & Privacy Considerations
- **Access Control**: Supabase Auth required; unauthenticated users see only the login panel.
- **Secret Hygiene**: Never commit `.env`; use Streamlit secrets in cloud.
- **Data Handling**: Contracts and embeddings remain in-memory; cleared on logout/session end.
- **Least Privilege**: Use Supabase anon key for client-side auth; avoid service-role keys.
- **Compliance Awareness**: Tool assists review; does not replace counsel. Keep PII and sensitive data secure.

### 10.1 Threat Model
- **Actors**: End users (authenticated), unauthorized visitors, cloud platform operators.
- **Assets**: Contract contents, authentication tokens, API keys (OpenAI, Supabase), embedding vectors.
- **Entry Points**: Streamlit UI, OAuth redirect endpoint, secrets storage, dependency supply chain.
- **Key Risks**:
  - Secrets exposure via misconfigured repo or unquoted TOML.
  - Token leakage through client-side logs or query params.
  - Model misuse (prompt injection) causing ungrounded answers.
  - Dependency tampering or incompatible updates breaking imports.

### 10.2 Mitigations
- **Secrets Hygiene**: `.env` excluded from VCS; Streamlit secrets only with quoted values; rotate keys periodically.
- **Token Handling**: Store only access/refresh tokens in `st.session_state`; avoid printing sensitive values; clear on logout.
- **OAuth Hardening**: Exact `SUPABASE_REDIRECT_URL` match; strip trailing slashes; use HTTPS in production.
- **Prompt Robustness**: Enforce citation requirement in prompts; instruct model to flag uncertainty and avoid speculation.
- **Supply Chain Control**: Pin `requirements.txt` and `runtime.txt`; change one variable at a time; maintain rollback.
- **Privacy by Design**: No persistent storage of contract text; in-memory processing; ephemeral notifications.

### 10.3 Compliance Notes
- **PII/Data Sensitivity**: Contracts may contain PII; restrict app access via Auth; consider data minimization and role-based views in future.
- **Logging Policy**: Avoid logging raw contract contents or tokens; retain only operational logs without sensitive payloads.
- **Data Residency**: Be aware of OpenAI/Supabase service regions; include notices if residency constraints apply.
- **User Consent**: Inform users that AI outputs are assistive, not legal advice; require acceptance of terms before use.

### 10.4 Incident Response
- **Detection**: Monitor redeploy logs and error surfaces for anomalies; add simple telemetry counters for auth failures.
- **Containment**: Revoke exposed keys; disable app temporarily via Streamlit dashboard.
- **Eradication**: Rotate credentials, audit dependencies, verify redirect URL alignment.
- **Recovery**: Redeploy from last known good commit; validate flows end-to-end; communicate status to users.

## 11. Testing & Evaluation
- **Functional Tests**:
  - PDF ingestion across varied contract formats.
  - Chunking correctness (clause boundaries preserved).
  - Retrieval effectiveness (top-k returns relevant sections).
  - RAG answers with citations present and accurate.
  - Voice transcription quality and error handling.
  - Auth flows: sign-up, sign-in, logout, Google OAuth redirect.
- **Performance Measurements**:
  - Embedding/cache timings: cold vs. warm runs.
  - Quick-action precompute latency vs. immediate availability.
- **User Feedback**:
  - UX clarity: full-width cards, minimal noise, consistent theming.
  - Value assessment: faster identification of obligations and risks.

### 11.1 Test Matrix
- **Functional**: Upload → Index → Chat → Citations; Voice record → Transcribe → Answer.
- **Auth**: Sign-up, sign-in, Google OAuth, logout, session restore.
- **Performance**: Cold vs warm index timings; quick-actions precompute latency; concurrent queries with `ThreadPoolExecutor`.
- **Resilience**: OpenAI quota exceeded → fallback embeddings; missing secrets → graceful errors; OAuth code absent → no session mutation.
- **Compatibility**: Python 3.11 runtime; Streamlit 1.51; `langchain 0.1.20` imports preserved.

### 11.2 Metrics & Instrumentation (Future)
- Add lightweight counters for: index build time, retrieval latency, answer token counts, quick-action completion times.
- Optional anonymized event logging (no contract text) for usability analysis.

## 12. Results
- **Reliability**: Stable builds achieved with pinned runtime and dependencies.
- **Usability**: Quick actions reduce cognitive overhead; voice adds convenience.
- **Accuracy**: Grounded responses improve trust; citations enable verification.
- **Maintainability**: Clean separation of concerns; secrets and config well-documented.

## 13. Limitations
- **OCR**: Image-only/scanned PDFs require OCR integration for full coverage.
- **Model Boundaries**: GPT-4o mini may need escalation for complex legal reasoning.
- **Single-Document Scope**: Current session focuses on one document; portfolio analytics are future work.
- **Citation Fidelity**: Section detection depends on input formatting; robust parsing can further improve.

## 14. Future Work & Roadmap
- **OCR + Layout Retention**: Integrate `pytesseract` and layout parsers for scanned documents.
- **Clause Intelligence**: Classification, normalization, and risk scoring per clause type.
- **Portfolio RAG**: Cross-document obligations and conflict detection across multiple contracts.
- **Persistence & Teams**: Encrypted index storage, user-specific history, and shared projects.
- **Observability**: Metrics for retrieval quality and latency; tracing for prompt health.
- **TTS Answers**: Add text-to-speech playback for hands-free review.

## 15. Conclusion
LegalLens demonstrates an effective blend of RAG, authentication, and voice interaction to streamline contract analysis. The system architecture emphasizes grounded responses, secure access, and practical cloud deployment. With a clear roadmap—OCR, clause intelligence, and portfolio analytics—LegalLens provides a robust foundation for next-generation legal-tech workflows.

## Appendix A: Configuration Cheatsheet
- `.env` (local):
  ```
  OPENAI_API_KEY=sk-...
  SUPABASE_URL=https://<project>.supabase.co
  SUPABASE_ANON_KEY=...
  SUPABASE_REDIRECT_URL=http://localhost:8501
  ```
- Streamlit secrets (cloud):
  ```toml
  OPENAI_API_KEY = "sk-..."
  SUPABASE_URL = "https://<project>.supabase.co"
  SUPABASE_ANON_KEY = "..."
  SUPABASE_REDIRECT_URL = "https://<app>.streamlit.app"
  ```
- Runtime pin: `runtime.txt` → `python-3.11.9`

## Appendix B: Dependency Pins
- `langchain==0.1.20`
- `langchain-openai==0.1.7`
- `langchain-community==0.0.38`
- `streamlit==1.51.0` (installed by Cloud)
- `faiss-cpu`, `pypdf`, `openai`, `python-dotenv`, `sentence-transformers`, `supabase`

## Appendix C: Operations Playbook
- Rebuild: Push to `master` → Redeploy in Streamlit Cloud.
- Common Errors:
  - ModuleNotFound: Ensure `langchain` and `langchain-community` pinned compatibly.
  - Python 3.13 conflicts: Add `runtime.txt` with `python-3.11.9`.
  - OAuth redirect mismatch: Align Streamlit URL and Supabase settings.
