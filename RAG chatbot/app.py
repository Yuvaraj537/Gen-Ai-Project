import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.schema import Document
import faiss
from PyPDF2 import PdfReader
#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

# -----------------------------
# Helper functions
# -----------------------------
def load_and_convert_document(file_path):
    """Extract text from PDF."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def get_markdown_splits(markdown_content):
    """Split markdown content by headers."""
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return splitter.split_text(markdown_content)

def setup_vector_store(chunks):
    """Create embeddings and FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS index
    single_vector = embeddings.embed_query("dummy")
    index = faiss.IndexFlatL2(len(single_vector))
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    # Ensure chunks are Documents
    docs = []
    for chunk in chunks:
        if isinstance(chunk, str):
            docs.append(Document(page_content=chunk))
        elif isinstance(chunk, Document):
            docs.append(chunk)
        else:
            docs.append(Document(page_content=str(chunk)))
    
    vector_store.add_documents(docs)
    return vector_store

def create_rag_chain(retriever, model_choice, api_key):
    """Create a RetrievalQA chain."""
    prompt_text = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Answer in bullet points and be concise.

Question: {question}
Context: {context}
Answer:
"""
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])

    if model_choice == "Groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=api_key)
    else:
        llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=api_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Amazon 10-Q RAG QA System", page_icon="📊", layout="wide")
st.title("🤖 LLM ChatBot")
st.markdown("Built with **Streamlit + LangChain + Groq/OpenAI**")

# Sidebar
st.sidebar.header("🔑 API Configuration")
model_choice = st.sidebar.selectbox("Choose Model Provider:", ["Groq", "OpenAI"], index=0)
api_key = st.sidebar.text_input("Enter your API key:")

if not api_key:
    st.warning("⚠️ Please enter your API key in the sidebar to start.")
    st.stop()

# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# -----------------------------
# Upload & Process PDF
# -----------------------------
st.header("Upload & Process Document")
uploaded_file = st.file_uploader("Upload your Amazon 10-Q PDF", type=["pdf"])

if uploaded_file is not None:
    if st.button("Upload & Process"):
        with st.spinner("Processing document... ⏳"):
            # Save uploaded PDF temporarily
            file_path = "uploaded_file.pdf"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load and split document
            markdown_content = load_and_convert_document(file_path)
            chunks = get_markdown_splits(markdown_content)
            st.session_state.vector_store = setup_vector_store(chunks)

            # Setup retriever and RAG chain
            retriever = st.session_state.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 3}
            )
            st.session_state.rag_chain = create_rag_chain(retriever, model_choice, api_key)

        st.success("✅ Document processed and RAG chain created! You can now ask questions.")

# -----------------------------
# Ask Questions
# -----------------------------
st.header("Ask Questions")
if st.session_state.rag_chain is None:
    st.info("Please upload and process a document first.")
else:
    question = st.text_input("Ask a Question:", placeholder="e.g. What was Amazon's total revenue in Q3 2024?")
    if st.button("🔍 Get Answer "):
        with st.spinner("Thinking... 🤖"):
            # Use call instead of run to get multiple outputs
            response = st.session_state.rag_chain({"query": question})
            answer = response["result"]
            sources = response["source_documents"]

            st.subheader("📢 Answer:")
            st.write(answer)

            st.subheader("📄 Retrieved Source Documents:")
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:500] + "...")  # show first 500 chars
