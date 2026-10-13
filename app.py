
import hashlib
import os
import re

import chromadb
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. APP SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Personal AI Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "personal_knowledge"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

st.title("📚 Personal AI Knowledge Assistant")
st.caption(
    "Your private document library, powered by semantic search "
    "and local AI."
)


# --------------------------------------------------
# 2. LOCAL PERSISTENT CHROMADB
# --------------------------------------------------

@st.cache_resource
def get_collection():
    os.makedirs(DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# Load the database, but do not load the embedding model yet.
try:
    collection = get_collection()
except Exception as exc:
    st.error(f"Could not open the local ChromaDB database: {exc}")
    st.stop()


# --------------------------------------------------
# 3. LOAD THE EMBEDDING MODEL ONLY WHEN NEEDED
# --------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def load_embedding_model():
    try:
        with st.spinner("Loading the embedding model..."):
            return get_embedding_model()
    except Exception as exc:
        st.error(f"Could not load the embedding model: {exc}")
        return None


# --------------------------------------------------
# 4. SPLIT TEXT INTO CHUNKS
# --------------------------------------------------

def split_text(text):
    text = re.sub(r"\s+", " ", text or "").strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# --------------------------------------------------
# 5. INDEX PDF PAGES AND PRESERVE PAGE NUMBERS
# --------------------------------------------------

def index_pdf(uploaded_file, embedding_model):
    pdf_bytes = uploaded_file.getvalue()
    filename = os.path.basename(uploaded_file.name)

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    reader = PdfReader(__import__("io").BytesIO(pdf_bytes))

    documents = []
    metadatas = []
    ids = []

    for page_index, page in enumerate(reader.pages):
        page_number = page_index + 1

        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""

        chunks = split_text(page_text)

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{file_hash}:{page_number}:{chunk_index}".encode("utf-8")
            ).hexdigest()

            documents.append(chunk)

            metadatas.append({
                "source": filename,
                "page": page_number,
                "chunk": chunk_index,
                "file_hash": file_hash,
            })

            ids.append(chunk_id)

    if not documents:
        return 0, (
            "No readable text was found. This PDF may be scanned "
            "and require OCR."
        )

    # Avoid indexing the exact same PDF repeatedly.
    existing = collection.get(
        where={"file_hash": file_hash},
        include=["metadatas"],
    )

    if existing["ids"]:
        return 0, f"{filename} is already indexed."

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(documents), None


# --------------------------------------------------
# 6. SHOW THE EXISTING KNOWLEDGE LIBRARY
# --------------------------------------------------

st.sidebar.header("📚 Knowledge Library")
st.sidebar.caption(
    "Indexed documents are stored locally in the chroma_db folder."
)

try:
    total_chunks = collection.count()
    stored = collection.get(include=["metadatas"])
    all_metadata = stored.get("metadatas") or []

    indexed_files = sorted({
        item.get("source", item.get("filename", "Unknown PDF"))
        for item in all_metadata
        if item
    })

    st.sidebar.metric("Indexed text chunks", total_chunks)

    st.sidebar.subheader("Indexed PDFs")

    if indexed_files:
        for filename in indexed_files:
            st.sidebar.write(f"📄 {filename}")
    else:
        st.sidebar.info("No documents indexed yet.")

except Exception as exc:
    st.sidebar.error(f"Could not read the library: {exc}")
    total_chunks = 0


st.sidebar.divider()
st.sidebar.caption(f"AI model: Ollama / {OLLAMA_MODEL}")
st.sidebar.caption("Embeddings: all-MiniLM-L6-v2")
st.sidebar.caption("Vector database: ChromaDB")


# --------------------------------------------------
# 7. ADD NEW DOCUMENTS WHEN YOU WANT
# --------------------------------------------------

st.header("1. Add documents to your library")
st.write("Select PDFs to add. Your existing indexed documents remain stored.")

uploaded_files = st.file_uploader(
    "Select one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_upload",
)

if st.button("📥 Index uploaded PDFs", disabled=not uploaded_files):
    embedding_model = load_embedding_model()

    if embedding_model is not None:
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    count, message = index_pdf(
                        uploaded_file,
                        embedding_model,
                    )

                if message:
                    st.warning(message)
                else:
                    st.success(
                        f"{uploaded_file.name}: indexed {count} text chunks."
                    )

            except Exception as exc:
                st.error(f"Could not index {uploaded_file.name}: {exc}")

        st.rerun()


# --------------------------------------------------
# 8. SEARCH THE EXISTING KNOWLEDGE LIBRARY
# --------------------------------------------------


def search_documents(question, embedding_model):
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K, total_chunks),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    excerpts = []
    seen = set()

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        metadata = metadata or {}

        source = metadata.get(
            "source", metadata.get("filename", "Unknown PDF")
        )
        page = metadata.get(
            "page", metadata.get("page_number", "Unknown")
        )

        # Avoid showing duplicate text.
        key = (source, page, document)
        if key in seen:
            continue
        seen.add(key)

        excerpts.append({
            "text": document,
            "source": source,
            "page": page,
            "distance": distance,
        })

    return excerpts



# --------------------------------------------------
# 9. ASK OLLAMA USING THE RETRIEVED EXCERPTS
# --------------------------------------------------


def ask_ollama(question, excerpts):
    context_parts = []

    for excerpt in excerpts:
        context_parts.append(
            f"Source: {excerpt['source']}, "
            f"Page: {excerpt['page']}\n"
            f"Text: {excerpt['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a research assistant that answers questions about PDF documents.

RULES:
1. Answer the user's question directly using the provided excerpts.
2. Combine relevant information from multiple excerpts when necessary.
3. Do not claim that an excerpt is missing when relevant information is present.
4. Do not invent facts that are not supported by the excerpts.
5. Cite factual claims using this format:
   [Source: research_paper_3.pdf, Page 2]
   Use the actual filename and page number provided in each excerpt.
6. If the excerpts do not contain enough relevant information, explain
   what information is missing and answer only what the excerpts support.
7. Use clear, simple language and organize the answer into bullet points
   when the question asks for findings or a summary.

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2
            },
        },
        timeout=(30, 600),
    )

    response.raise_for_status()

    answer = response.json().get("response", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    return answer


# --------------------------------------------------
# 10. CHAT WITH YOUR DOCUMENTS
# --------------------------------------------------

st.divider()
st.header("💬 Chat with your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("📖 View source pages"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']} — Page {source['page']}**"
                    )
                    st.write(source["text"])


question = st.chat_input("Ask a question about your indexed PDFs...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if collection.count() == 0:
            answer = (
                "Your local knowledge library is empty. "
                "Please add and index a PDF first."
            )
            sources = []

        else:
            embedding_model = load_embedding_model()

            if embedding_model is None:
                answer = (
                    "The embedding model could not be loaded. "
                    "Please check the error above."
                )
                sources = []

            else:
                try:
                    with st.spinner("Searching your documents..."):
                        excerpts = search_documents(
                            question,
                            embedding_model,
                        )

                    if not excerpts:
                        answer = "I could not find relevant excerpts in your library."
                        sources = []

                    else:
                        with st.spinner("Generating an answer with Ollama..."):
                            answer = ask_ollama(question, excerpts)

                        sources = excerpts

                except requests.exceptions.ConnectionError:
                    answer = (
                        "Could not connect to Ollama. Make sure Ollama is "
                        "running and the llama3.2 model is installed."
                    )
                    sources = []

                except requests.exceptions.Timeout:
                    answer = (
                        "Ollama took too long to respond. Try a shorter "
                        "question or check whether the model is still running."
                    )
                    sources = []

                except Exception as exc:
                    answer = f"An error occurred: {exc}"
                    sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📖 View retrieved source pages"):
                seen = set()

                for source in sources:
                    key = (source["source"], source["page"])

                    if key in seen:
                        continue

                    seen.add(key)

                    st.markdown(
                        f"**{source['source']} — Page {source['page']}**"
                    )
                    st.write(source["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })


# --------------------------------------------------
# 11. CLEAR THE CHAT WITHOUT DELETING DOCUMENTS
# --------------------------------------------------

if st.sidebar.button("🗑️ Clear chat"):
    st.session_state.messages = []
    st.rerun()
