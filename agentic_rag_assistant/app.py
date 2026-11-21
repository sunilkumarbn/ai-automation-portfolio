import os
import shutil
from pathlib import Path
import fitz
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# --- Paths & Local Models ---
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "vector_store"
TMP_INDEX_DIR = BASE_DIR / "vector_store_tmp"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)


def _prepare_clean_dir(target: Path) -> None:
    """Remove existing file/folder at target cleanly before creating a directory."""
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
    elif target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
        if target.exists():
            # Fallback if a nested file is locked on Windows
            for root, dirs, files in os.walk(target, topdown=False):
                for name in files:
                    p = Path(root) / name
                    try:
                        p.chmod(0o666)
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                for name in dirs:
                    try:
                        (Path(root) / name).rmdir()
                    except OSError:
                        pass
            try:
                target.rmdir()
            except OSError:
                pass
    target.mkdir(parents=True, exist_ok=True)


# --- 1. Document Ingestion & FAISS Indexing ---
def build_vector_store(uploaded_files) -> FAISS:
    docs: list[Document] = []

    for file in uploaded_files:
        file_bytes = file.getvalue()
        if file.name.lower().endswith(".pdf"):
            with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
                for page_num, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": file.name, "page": page_num},
                            )
                        )
        else:
            text = file_bytes.decode("utf-8", errors="ignore").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": file.name, "page": 1},
                    )
                )

    if not docs:
        raise ValueError("No readable text could be extracted from the uploaded files.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save to temp directory first, then replace INDEX_DIR cleanly
    _prepare_clean_dir(TMP_INDEX_DIR)
    vectorstore.save_local(str(TMP_INDEX_DIR))

    if INDEX_DIR.is_symlink() or INDEX_DIR.is_file():
        INDEX_DIR.unlink(missing_ok=True)
    elif INDEX_DIR.is_dir():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)

    if not INDEX_DIR.exists():
        TMP_INDEX_DIR.replace(INDEX_DIR)
    else:
        # Copy into existing directory if Windows held the top folder handle
        shutil.copytree(TMP_INDEX_DIR, INDEX_DIR, dirs_exist_ok=True)
        shutil.rmtree(TMP_INDEX_DIR, ignore_errors=True)

    return vectorstore


def get_vector_store() -> FAISS | None:
    if "vectorstore" in st.session_state and st.session_state["vectorstore"] is not None:
        return st.session_state["vectorstore"]

    if (INDEX_DIR / "index.faiss").exists():
        vs = FAISS.load_local(
            str(INDEX_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        st.session_state["vectorstore"] = vs
        return vs

    return None


# --- 2. Agent Tools ---
@tool
def search_knowledge_base(query: str) -> str:
    """Search indexed documents and PDFs for relevant facts with source and page metadata."""
    vs = get_vector_store()
    if vs is None:
        return "Knowledge base is empty. Ask the user to upload and index documents first."

    results = vs.similarity_search(query, k=4)
    if not results:
        return "No relevant passages found in the indexed documents."

    formatted = []
    for doc in results:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[{src}, Page {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


@tool
def python_calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression such as '(1250 * 0.18) + 1250'."""
    allowed = set("0123456789+-*/().% ")
    if not expression or not all(ch in allowed for ch in expression):
        return "Invalid mathematical expression."
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as exc:
        return f"Calculation error: {exc}"


@st.cache_resource
def load_agent():
    return create_react_agent(
        llm,
        tools=[search_knowledge_base, python_calculator],
        prompt=(
            "You are a precise Technical & Document Assistant. "
            "Always call `search_knowledge_base` when answering questions about uploaded documents, "
            "and cite exact [Source, Page] references from the returned chunks. "
            "Use `python_calculator` for arithmetic. If the answer is not in the documents, state that clearly."
        ),
    )


agent = load_agent()

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Agentic RAG Assistant", layout="centered")
st.title("Agentic RAG & Tool-Calling Assistant")

with st.sidebar:
    st.header("Document Ingestion")
    files = st.file_uploader(
        "Upload PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index Documents", type="primary"):
        if not files:
            st.warning("Please upload at least one file first.")
        else:
            try:
                with st.spinner("Chunking & building FAISS index..."):
                    st.session_state["vectorstore"] = build_vector_store(files)
                st.success(f"Indexed {len(files)} file(s) into FAISS!")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question about your documents or run a calculation..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Agent reasoning & retrieving..."):
            response = agent.invoke({"messages": [("user", user_input)]})
            answer = response["messages"][-1].content
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})