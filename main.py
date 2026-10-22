
"""RAG pipeline for Nimbus AI customer support."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "customer_support.pdf"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

FALLBACK_ANSWER = (
    "I could not find that in our support documentation. "
    "Please contact our support team for further help."
)


def build_vectorstore():
    """Load the support PDF and create a FAISS search index."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"Support PDF not found: {PDF_PATH}. "
            "Place customer_support.pdf in the project folder."
        )

    documents = PyPDFLoader(str(PDF_PATH)).load()

    if not documents:
        raise ValueError(
            "No readable text was found in customer_support.pdf."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.from_documents(chunks, embeddings)


def answer_question(question: str, vectorstore) -> tuple[str, list]:
    """Retrieve relevant handbook content and generate an answer."""
    question = question.strip()

    if not question:
        return "Please enter a question.", []

    retrieved_docs = vectorstore.similarity_search(question, k=3)

    if not retrieved_docs:
        return FALLBACK_ANSWER, []

    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    prompt = f"""You are NimbusCart's customer support assistant.
Answer the customer's question using ONLY the context below.
Do not invent policies, dates, fees, or procedures.
If the answer is not available in the context, reply exactly:
{FALLBACK_ANSWER}

Context:
{context}

Question: {question}

Answer:"""

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
    )

    response = llm.invoke(prompt)

    return response.content, retrieved_docs
