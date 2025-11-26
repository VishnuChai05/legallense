# ⚖️ LegalLens - AI-Powered Multimodal Contract Analyzer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**LegalLens** is an advanced multimodal AI application that revolutionizes contract analysis by combining RAG (Retrieval-Augmented Generation) technology with voice-enabled interactions. Built with Streamlit and powered by OpenAI's GPT-4o mini family, it enables users to upload legal contracts and interact with them through both text-based chat and voice queries.

## 🚀 Features

- **📄 PDF Contract Upload**: Support for multi-page legal documents
- **💬 RAG-Powered Text Chat**: Ask questions and get precise answers with section citations
- **🎤 Voice Question & Answer**: Speak naturally to query your contracts using multimodal AI
- **⚠️ Automatic Red Flag Detection**: AI identifies potential legal risks and concerning clauses
- **🔍 Smart Contract Summarization**: Instant summaries of parties, obligations, and key terms
- **📅 Key Date Extraction**: Automatically finds deadlines and time-sensitive obligations
 - **🔐 Supabase Auth**: Email/password sign-in plus "Continue with Google"
- **🔄 Resilient Embeddings**: Automatically falls back to local sentence-transformer embeddings if OpenAI quotas are hit
- **🎨 Professional Legal Tech UI**: Dark blue theme optimized for legal professionals
- **☁️ Cloud-Ready**: Deployed on Streamlit Community Cloud with secure secrets management

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI Model**: OpenAI GPT-4o mini (text + reasoning)
- **Embeddings**: OpenAI Text Embedding 3 Large (with local `sentence-transformers/all-MiniLM-L6-v2` fallback)
- **Vector Database**: FAISS (CPU version)
- **Framework**: LangChain
- **Speech-to-Text**: OpenAI GPT-4o mini Transcribe
- **PDF Processing**: PyPDF
- **Deployment**: Streamlit Community Cloud

## 📋 Prerequisites

- Python 3.9 or higher
- OpenAI API Key (with access to GPT-4o mini + embeddings + transcription)
- Supabase project (Auth enabled with email + Google providers)
- Git (for version control)

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/legallens.git
cd legallens
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Key (Local Development)

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

**To get your OpenAI API Key:**
1. Visit [OpenAI Platform](https://platform.openai.com/account/api-keys)
2. Sign in with your OpenAI account (create one if needed)
3. Click "Create new secret key"
4. Copy the key and paste it in your `.env` file

### 4. Configure Supabase Auth

LegalLens now requires Supabase for authentication. After creating your Supabase project and enabling email + Google providers, add the following to your local `.env` (and Streamlit secrets when deploying):

```bash
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_ANON_KEY=<your-anon-key>
SUPABASE_REDIRECT_URL=http://localhost:8501
OPENAI_API_KEY=your_openai_api_key_here
```

Set `SUPABASE_REDIRECT_URL` to match the redirect URL configured in Supabase Auth settings. When deploying, replace it with your public Streamlit URL.

### 5. Run Locally

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## ☁️ Deployment Instructions

### Deploy to Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit - LegalLens project"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch (`main`), and main file (`app.py`)

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, click on your app
   - Go to "Settings" → "Secrets"
   - Add the following in TOML format:
   ```toml
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

4. **Deploy**
   - Click "Deploy!"
   - Your app will be live at `https://your-app-name.streamlit.app`

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
- **Summarize Contract**: Get instant overview of the document
- **Identify Red Flags**: Detect potential legal risks
- **Extract Key Dates**: Find all important deadlines

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
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── .env                  # Local environment variables (not committed)
```

## 🐛 Troubleshooting

### Issue: "OpenAI API Key not found"
**Solution**: Ensure your `.env` file exists locally or secrets are configured in Streamlit Cloud.

### Issue: "Failed to create embeddings with OpenAI"
**Solution**: This usually means your OpenAI account hit an embedding quota. LegalLens will automatically switch to local sentence-transformer embeddings; expect slightly slower performance. You can monitor usage at <https://platform.openai.com/usage>.

### Issue: "Module not found" errors
**Solution**: Run `pip install -r requirements.txt` to install all dependencies.

### Issue: Audio not working
**Solution**: Ensure you're using a modern browser (Chrome/Edge recommended) and have granted microphone permissions.

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

## 👨‍💻 Author

**Your Name**  
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- OpenAI team for GPT-4o mini and transcription capabilities
- Streamlit team for the excellent web framework
- LangChain community for RAG infrastructure
- FAISS team for efficient vector similarity search

## 📊 Project Status

✅ **Ready for Submission** - All core features implemented and tested.

---

**Built with ❤️ for the Gen AI Capstone Project**
