# ⚖️ LegalLens – AI-Powered Multimodal Contract Analyzer

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.51-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)

**LegalLens** is an AI assistant for legal teams that blends Retrieval-Augmented Generation (RAG), Supabase-backed authentication, and voice interactions. Users can upload contracts, interrogate them through chat or speech, and receive curated quick actions such as summaries, key dates, or risk highlights.

## 🚀 Features

- **📄 PDF Contract Upload**: Handles lengthy, multi-page legal documents.
- **💬 RAG Chat Workspace**: Chat with your contract and receive grounded answers with section call-outs.
- **🎤 Voice Q&A**: Record questions using Google Speech Recognition (free) or OpenAI Whisper.
- **⚠️ Risk Score Analysis**: AI-powered risk scoring (0-100) with confidence levels and identified risks.
- **📅 Key Date Extraction**: Pulls deadlines and time-sensitive obligations on demand.
- **✨ Quick Actions**: Precomputes summaries, red flags, and date lists as soon as a document finishes indexing.
- **🔐 Supabase Auth**: Email/password plus Google OAuth with session caching inside Streamlit.
- **🤖 Multi-LLM Support**: Works with OpenAI GPT-4o, **Ollama local models** (llama3.2, tinyllama), or Groq.
- **🔄 Resilient Embeddings**: Falls back to local `all-MiniLM-L6-v2` embeddings when OpenAI quotas are exhausted.
- **🚀 Cloud Ready**: Optimized for Streamlit Community Cloud with cached vector stores and pinned runtimes.

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit 1.51 with custom legal-tech theme |
| **Backend** | FastAPI server with /chat, /risk, /upload endpoints |
| **Orchestration** | LangChain for text splitting, vector search, and RAG chains |
| **LLMs** | OpenAI GPT-4o mini, **Ollama** (local), or Groq (cloud) |
| **Speech** | Google Speech Recognition (free) / OpenAI Whisper |
| **Embeddings** | OpenAI Text Embedding 3 Large with fallback to HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (CPU) cached with `st.cache_resource` |
| **Authentication** | Supabase Auth (email/password + Google OAuth) |
| **Deployment** | Streamlit Community Cloud (Python 3.11 runtime) |

## 📋 Prerequisites

- Python 3.11+ (matches the pinned Streamlit Cloud runtime)
- **One of the following LLM options:**
  - OpenAI API key with access to GPT-4o mini
  - Ollama installed locally (for offline/free usage)
  - Groq API key (free tier available)
- Supabase project with Auth enabled for Email and Google providers
- GitHub repository (required for Streamlit Community Cloud deployment)

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/legallens.git
cd legallens
```

### 2. Create & Activate a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (Local Development)

Create a `.env` file in the project root containing:

```bash
# LLM Configuration (choose one)
OPENAI_API_KEY=your_openai_api_key_here    # For OpenAI
USE_OPENAI=true                             # Set to "false" to use local Ollama

# Ollama Configuration (for local LLM - free, offline)
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.2:3b                    # or tinyllama, llama3.1:8b

# Supabase Authentication
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_ANON_KEY=<your-anon-key>
SUPABASE_REDIRECT_URL=http://localhost:8501
```

> `SUPABASE_REDIRECT_URL` must exactly match the redirect URL configured in Supabase Auth settings.

### 5. (Optional) Set Up Ollama for Local LLM

If you want to run without OpenAI (free, offline):

```bash
# Install Ollama from https://ollama.ai
# Then pull a model:
ollama pull llama3.2:3b      # Recommended (balanced speed/quality)
ollama pull tinyllama        # Faster but lower quality
ollama pull llama3.1:8b      # Better quality but slower on CPU
```

### 6. Run Locally

**With OpenAI:**
```bash
streamlit run app.py
```

**With Ollama (local, free):**
```bash
# PowerShell
$env:USE_OPENAI="false"; $env:OLLAMA_URL="http://127.0.0.1:11434"; $env:OLLAMA_MODEL="llama3.2:3b"; streamlit run app.py

# Bash
USE_OPENAI=false OLLAMA_URL=http://127.0.0.1:11434 OLLAMA_MODEL=llama3.2:3b streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## ☁️ Deployment Instructions

### Deploy to Streamlit Community Cloud

1. **Verify Runtime Pin**
   - Ensure `runtime.txt` contains `python-3.11.9`. Streamlit Cloud will create a Python 3.11 environment automatically.

2. **Push to GitHub** (main branch is `master` in this project)
   ```bash
   git add .
   git commit -m "Deploy LegalLens"
   git push origin master
   ```

3. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch (`main`), and main file (`app.py`)

4. **Configure Secrets**
   - In Streamlit Cloud dashboard, click on your app
   - Go to "Settings" → "Secrets"
   - Add the following in TOML format:
   ```toml
   OPENAI_API_KEY = "sk-..."
   SUPABASE_URL = "https://<your-project>.supabase.co"
   SUPABASE_ANON_KEY = "<your-anon-key>"
   SUPABASE_REDIRECT_URL = "https://<your-app-name>.streamlit.app"
   ```
   - Include the quotes (`"value"`) or Streamlit will reject the secrets.
   - Confirm that the redirect URL matches the one set in Supabase Auth → URL Configuration.

5. **Deploy / Redeploy**
   - Click "Deploy!"
   - Your app will be live at `https://your-app-name.streamlit.app`
    - To pick up new changes later, push to GitHub and choose "Redeploy" from the app dashboard.

## 📖 How to Use

### Sign In
1. Open the app and authenticate with your email/password (or use **Continue with Google**).
2. If you just created an account and email confirmations are enabled, complete the verification sent by Supabase.

### Text Chat Mode
1. Upload a PDF contract via the sidebar
2. Navigate to the "Chat" tab
3. Type your questions in the chat input
4. Receive AI-generated answers with section citations

### Voice Mode
1. Upload a PDF contract via the sidebar
2. Navigate to the "Voice Mode" tab
3. Click the microphone icon and speak your question
4. Click "Process Voice Question" to get your answer
### Dashboard Quick Actions
- **Summarize Contract**: Precomputed overview of the parties and obligations.
- **Identify Red Flags**: Cached analysis of risky or unusual clauses.
- **Extract Key Dates**: Lists deadlines and renewal windows referenced in the document.

Quick actions refresh any time you upload a new document; results are cached to avoid recomputation during the session.

## 🔒 Security & Privacy

- API keys are securely managed through Streamlit secrets
- No contract data is stored permanently
- All processing happens in-memory during the session
- Uses environment variables for local development
- Authentication sessions are issued by Supabase Auth (email/password + Google)

## 📂 Project Structure

```
legallens/
├── app.py                 # Main Streamlit application
├── backend/
│   ├── __main__.py        # Backend entry point
│   └── app.py             # FastAPI server with /chat, /risk, /upload endpoints
├── legal_scraper/         # Legal gazette data collection module
│   ├── fetch.py           # Web scraping utilities
│   ├── normalizer.py      # Text normalization
│   ├── exporter.py        # Data export functions
│   └── sources.yaml       # Scraping source configuration
├── tests/                 # Comprehensive test suite
│   ├── test_backend_routes.py
│   ├── test_risk_confidence.py
│   ├── test_voice_and_chat.py
│   └── ...
├── data/
│   ├── raw/               # Scraped legal documents (HTML, PDF)
│   ├── exports/           # Processed JSONL exports
│   └── vector/            # FAISS vector indices
├── .streamlit/
│   ├── config.toml        # Streamlit theme configuration
│   └── secrets.toml.example  # Secrets template for deployment
├── requirements.txt       # Python dependencies
├── runtime.txt            # Streamlit Cloud runtime pin (Python 3.11.9)
├── pytest.ini             # Test configuration
└── README.md              # Documentation (you are here)
```

## 🐛 Troubleshooting

### Issue: "OpenAI API Key not found"
**Solution**: Ensure your `.env` file exists locally or secrets are configured in Streamlit Cloud. Alternatively, use Ollama for local LLM.

### Issue: "OpenAI quota exceeded (429 error)"
**Solution**: LegalLens automatically falls back to local embeddings. For chat/risk features, set up Ollama:
```bash
ollama pull llama3.2:3b
# Then run with USE_OPENAI=false
```

### Issue: "Failed to create embeddings with OpenAI"
**Solution**: This usually means your OpenAI account hit an embedding quota. LegalLens will automatically switch to local sentence-transformer embeddings; expect slightly slower performance.

### Issue: "Ollama responses are slow"
**Solution**: Local LLM inference on CPU takes 30-60+ seconds. Options:
- Use a smaller model: `ollama pull tinyllama`
- Enable GPU acceleration if you have NVIDIA GPU
- Use Groq (free cloud API) for faster responses

### Issue: "Module not found" errors
**Solution**: Run `pip install -r requirements.txt` to install all dependencies.

### Issue: Audio/Voice not working
**Solution**: 
- Ensure you're using a modern browser (Chrome/Edge recommended)
- Grant microphone permissions
- Voice uses Google Speech Recognition (free) by default when OpenAI is disabled

### Issue: FAISS import errors
**Solution**: Make sure you're using `faiss-cpu` (not `faiss-gpu`) as specified in requirements.txt.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Maintainers

- Project Lead: *Update with your name / team*
- Issues & support: Please open a GitHub issue with detailed reproduction steps.

## 🙏 Acknowledgments

- OpenAI for GPT-4o mini text + audio capabilities.
- **Ollama** for enabling free, local LLM inference.
- **Google** for free Speech Recognition API.
- Streamlit for the rapid prototyping framework.
- LangChain & FAISS communities for the RAG tooling ecosystem.
- Supabase for authentication services.

---

**Built with ❤️ for the Gen AI Capstone Project**

**Repository**: [github.com/VishnuChai05/legallense](https://github.com/VishnuChai05/legallense)
