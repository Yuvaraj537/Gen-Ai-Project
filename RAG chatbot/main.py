import streamlit as st
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import faiss
from PyPDF2 import PdfReader

# -----------------------------
# Helper Functions
# -----------------------------

def load_and_convert_document(file_path):
    """Extract text from PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text


def get_markdown_splits(text):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )
    return splitter.split_text(text)


def setup_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS index
    dim = len(embeddings.embed_query("test"))
    index = faiss.IndexFlatL2(dim)

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    docs = []
    for chunk in chunks:
        if isinstance(chunk, Document):
            docs.append(chunk)
        else:
            docs.append(Document(page_content=str(chunk)))

    vector_store.add_documents(docs)
    return vector_store


def create_rag_chain(retriever, model_choice, api_key):
    prompt_template = """
You are a helpful assistant.
Use the context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer (bullet points):
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    if model_choice == "Groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            api_key=api_key
        )
    else:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key
        )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Chatbot (FAISS + LangChain)")
st.markdown("Upload a PDF and ask questions using LLMs")

# Sidebar
st.sidebar.header("🔐 API Settings")
model_choice = st.sidebar.selectbox("Choose LLM", ["Groq", "OpenAI"])
api_key = st.sidebar.text_input("API Key", type="password")

if not api_key:
    st.warning("Please enter your API key")
    st.stop()

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# -----------------------------
# Upload PDF
# -----------------------------

st.header("📄 Upload PDF")
uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file:
    if st.button("Process Document"):
        with st.spinner("Processing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())

            text = load_and_convert_document("temp.pdf")
            chunks = get_markdown_splits(text)
            st.session_state.vector_store = setup_vector_store(chunks)

            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 3}
            )

            st.session_state.rag_chain = create_rag_chain(
                retriever,
                model_choice,
                api_key
            )

        st.success("Document indexed successfully!")

# -----------------------------
# Ask Questions
# -----------------------------

st.header("❓ Ask Questions")

if st.session_state.rag_chain:
    query = st.text_input("Enter your question")
    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain({"query": query})

            st.subheader("📌 Answer")
            st.write(response["result"])

            st.subheader("📚 Source Documents")
            for i, doc in enumerate(response["source_documents"], 1):
                st.markdown(f"**Source {i}**")
                st.write(doc.page_content[:500] + "...")
else:
    st.info("Upload and process a document first.")
