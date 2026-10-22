# NimbusCart AI — RAG-Powered Customer Support Assistant

A local Retrieval-Augmented Generation (RAG) chatbot that answers customer-support questions using the NimbusCart Customer Support Policy Handbook PDF.

## Tech stack
- Python and Streamlit
- LangChain document loading and text splitting
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings
- FAISS vector similarity search
- Ollama running the `llama3` model

## Requirements
- Python 3.10 or 3.11 recommended
- Ollama installed and running locally
- The `llama3` model downloaded in Ollama

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3
ollama serve
```

If Ollama is already running as a desktop/background service, do not start a second `ollama serve` process. Open another terminal in the project folder and run:

```powershell
streamlit run app.py
```

The first run may download the embedding model and take a little while to build the FAISS index from the PDF.

## Project layout

```text
NimbusCart-AI/
├── app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── customer_support.pdf
├── data/
└── artifacts/
```

## Notes
- The chatbot should answer only from the supplied handbook and use a fallback response when the information is not found.
- This version builds the FAISS index at app startup and caches it for the Streamlit session. The `artifacts/` folder is reserved for a future persisted index.
- Do not commit `.venv/`, secrets, or large generated model files to Git.
