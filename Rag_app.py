import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Load Environment Variables from .env file ---
load_dotenv()

# --- Page Setup ---
st.set_page_config(
    page_title="UAE Policy RAG Assistant",
    page_icon="🇦🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Lively Atmosphere ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .uae-header {
        background: linear-gradient(90deg, #007A3D 0%, #000000 35%, #FF0000 100%);
        padding: 4px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    .uae-header-inner {
        background-color: #ffffff;
        padding: 20px 30px;
        border-radius: 8px;
        text-align: center;
    }
    .uae-title {
        color: #007A3D;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
    }
    .uae-subtitle {
        color: #475569;
        font-size: 1.05rem;
        margin-top: 5px;
    }
    .metric-card {
        background-color: #ffffff;
        border-left: 5px solid #007A3D;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    div[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Secure Credentials Retrieval ---
# Pulls from .env file or environment variables
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# --- Lively Header ---
st.markdown("""
    <div class="uae-header">
        <div class="uae-header-inner">
            <h1 class="uae-title">🇦🇪 UAE Policy Intelligence Assistant</h1>
            <p class="uae-subtitle">Instant, compliant, and verified policy answers powered by AI Vector Retrieval</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Top Stats Bar ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-value">🇦🇪 UAE</div><div class="metric-label">Policy Framework</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-value">⚡ Fast</div><div class="metric-label">Vector Search</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">🛡️ Active</div><div class="metric-label">Guardrail Security</div></div>', unsafe_allow_html=True)
with col4:
    status_text = "Ready" if "vectorstore" in st.session_state else "Pending Index"
    st.markdown(f'<div class="metric-card"><div class="metric-value">{status_text}</div><div class="metric-label">Database Status</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("📄 Document Setup")
    pdf_path = st.text_input("Policy Document Path", value="KG_RAG_project/KG Rag doc.pdf")

    if st.button("🚀 Process & Index Document", type="primary", use_container_width=True):
        if not openai_api_key:
            st.error("OpenAI API Key missing. Make sure it is defined in your .env file.")
        elif not os.path.exists(pdf_path):
            st.error(f"File '{pdf_path}' not found. Check path.")
        else:
            with st.spinner("Processing document chunks & creating vector index..."):
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(docs)

                    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
                    vectorstore = FAISS.from_documents(chunks, embeddings)

                    st.session_state["vectorstore"] = vectorstore
                    st.success(f"Indexed {len(docs)} pages into {len(chunks)} searchable chunks!")
                    st.rerun()
                except Exception as e:
                    st.error("Ingestion failed:")
                    st.exception(e)

    st.markdown("---")
    st.markdown("### 🏛️ Pipeline Features")
    st.write("✨ **Vector Search:** FAISS Engine")
    st.write("✨ **Embeddings:** OpenAI Embeddings")
    st.write("✨ **Compliance:** Input & Output Guardrails")

# --- Guardrail Rules ---
def input_guardrail(query: str) -> bool:
    prohibited_terms = ["hack", "exploit", "bypass", "malware", "system prompt", "jailbreak"]
    query_lower = query.lower()
    for term in prohibited_terms:
        if term in query_lower:
            return False
    return True

def output_guardrail(answer: str) -> str:
    if not answer or len(answer.strip()) < 10:
        return "I could not verify this information within the official policy document."
    return answer

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Query Interface ---
st.subheader("💬 Query the Policy")
question = st.text_area(
    "Enter your question",
    placeholder="Example: What are the key compliance rules outlined in the policy?",
    height=110,
)

if st.button("🔍 Search Policy Information", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    elif not openai_api_key:
        st.error("OpenAI API Key is required to run queries. Check your .env file.")
    elif "vectorstore" not in st.session_state:
        st.warning("Please index a document using the sidebar first.")
    elif not input_guardrail(question):
        st.error("⚠️ Input Guardrail Triggered: Restricted words or disallowed prompt detected.")
    else:
        with st.spinner("Searching document context and generating response..."):
            try:
                vectorstore = st.session_state["vectorstore"]
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                retrieved_docs = retriever.invoke(question)

                prompt = ChatPromptTemplate.from_template(
                    """You are an official AI assistant for policy analysis.
                    Answer the user's question using ONLY the retrieved context below.
                    If the answer cannot be determined from the context, clearly state that the information is unavailable.

                    Context:
                    {context}

                    Question: {question}
                    """
                )

                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_api_key)

                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                raw_answer = chain.invoke(question)
                final_answer = output_guardrail(raw_answer)

                st.markdown("### 📋 Verified Answer")
                st.info(final_answer)

                st.markdown("### 📑 Reference Context Chunks")
                for i, doc in enumerate(retrieved_docs, start=1):
                    page_num = doc.metadata.get('page', 'N/A')
                    display_page = page_num + 1 if isinstance(page_num, int) else page_num
                    with st.expander(f"Reference Chunk #{i} (Page {display_page})"):
                        st.write(doc.page_content)

            except Exception as exc:
                st.error("An error occurred during query execution.")
                st.exception(exc)