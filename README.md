# Knowledgebase Q&A — RAG

A complete Python RAG application for asking questions over CSV knowledgebase.

## Requirements

- Python 3.10+
- An Google API key

## Setup

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```env
GOOGLE_API_KEY=your_api_key_here
```

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Web UI

```bash
streamlit run main.py
```

Open the URL Streamlit prints, upload documents, click **Build Knowledgebase**, then ask questions.

## RAG behavior

The bot:

- splits documents into chunks;
- creates embeddings ;
- stores normalized vectors in a local FAISS index;
- retrieves the most relevant chunks;
- sends only retrieved context to the chat model;
- instructs the model not to invent information;
- returns the retrieved source chunks for inspection.
