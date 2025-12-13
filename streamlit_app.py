import streamlit as st
import os

# Mock imports for type hints (won't be used if packages aren't installed)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_community.llms import Ollama
    from langchain.chains import RetrievalQA
    PACKAGES_AVAILABLE = True
except ImportError:
    PACKAGES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SEC Filing Summarizer & Q&A",
    page_icon="🏛️",
    layout="centered"
)

# Custom CSS with dark finance theme and subtle animations
st.markdown("""
<style>
/* Dark finance theme */
body {
    background-color: #0c1424;
    color: #e0e0e0;
    font-family: 'Arial', sans-serif;
}

/* Animated background elements */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-20px); }
    100% { transform: translateY(0px); }
}

@keyframes pulse {
    0% { opacity: 0.05; }
    50% { opacity: 0.1; }
    100% { opacity: 0.05; }
}

.background-animation {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    overflow: hidden;
}

.background-circle {
    position: absolute;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(41, 128, 185, 0.2) 0%, transparent 70%);
    opacity: 0.05;
    animation: float 15s infinite ease-in-out;
}

.background-line {
    position: absolute;
    height: 1px;
    background: linear-gradient(to right, transparent, #2980b9, transparent);
    opacity: 0.1;
    animation: pulse 8s infinite linear;
}

/* Main container */
.main-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 2rem;
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    box-shadow: 0 8px 32px rgba(2, 12, 30, 0.6);
    border: 1px solid rgba(94, 129, 172, 0.3);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* Title styling */
.title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #3498db, #2c3e50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #94a3b8;
    margin-bottom: 2rem;
}

/* Input styling */
.stTextInput > div > div > input {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #e0e0e0 !important;
    padding: 1rem !important;
    font-size: 1.1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #3498db !important;
    box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2) !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #2980b9, #2c3e50) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Answer box styling */
.answer-box {
    background: rgba(30, 41, 59, 0.7);
    border-left: 4px solid #3498db;
    padding: 1.5rem;
    border-radius: 0 10px 10px 0;
    margin-top: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.source-box {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
}

.source-header {
    font-weight: 600;
    color: #3498db;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.source-content {
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #cbd5e1;
}

/* Loading spinner */
.stSpinner > div {
    color: #3498db !important;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #334155;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-container {
        padding: 1rem;
        margin: 1rem;
    }
    
    .title {
        font-size: 2rem;
    }
    
    .subtitle {
        font-size: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Create animated background elements
st.markdown("""
<div class="background-animation">
    <div class="background-circle" style="width: 300px; height: 300px; top: 10%; left: 5%; animation-delay: 0s;"></div>
    <div class="background-circle" style="width: 200px; height: 200px; top: 60%; left: 80%; animation-delay: -5s;"></div>
    <div class="background-circle" style="width: 150px; height: 150px; top: 30%; left: 70%; animation-delay: -10s;"></div>
    <div class="background-line" style="width: 200px; top: 20%; left: 20%; animation-delay: 0s;"></div>
    <div class="background-line" style="width: 300px; top: 70%; left: 50%; animation-delay: -3s;"></div>
</div>
""", unsafe_allow_html=True)

# Main container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# Title and subtitle
st.markdown("<h1 class='title'>SEC Filing Summarizer & Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions about 10-K / 10-Q filings with citations</p>", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.qa_chain = None

@st.cache_resource
def initialize_rag():
    """Initialize the RAG components"""
    if not PACKAGES_AVAILABLE:
        st.warning("Required packages not installed. Please install langchain, langchain-community, chromadb, and sentence-transformers.")
        return None, False
        
    try:
        DB_PATH = "chroma_db"
        
        # Check if database exists
        if not os.path.exists(DB_PATH):
            st.error("Chroma database not found. Please run the ingestion script first.")
            return None, False
        
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load vector store
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Local LLM (make sure Ollama is running with llama3 model)
        try:
            llm = Ollama(model="llama3")
            # Test if model is available
            llm("Hello")
        except Exception as e:
            st.warning("LLM not available. Using a mock response instead.")
            llm = None
        
        # RAG Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain, True
    except Exception as e:
        st.error(f"Error initializing RAG components: {str(e)}")
        return None, False

# Initialize components
if not st.session_state.initialized:
    with st.spinner("Initializing RAG components..."):
        st.session_state.qa_chain, success = initialize_rag()
        st.session_state.initialized = True
        
        if not success:
            st.stop()

# Question input
question = st.text_input("", placeholder="Enter your question about SEC filings...", label_visibility="collapsed")

# Ask button
ask_button = st.button("Ask", type="primary")

# Process question
if ask_button and question:
    if not PACKAGES_AVAILABLE:
        # Mock response when packages aren't available
        st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
        st.markdown("**Answer:** This is a mock response. To get real answers, please install the required packages and run the ingestion script.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Mock sources
        st.markdown("### Sources / Citations")
        with st.expander("Source 1: sec_filings.csv"):
            st.markdown("<div class='source-content'>This is mock source content. In a real implementation, this would show the relevant excerpt from the SEC filing that was used to generate the answer.</div>", unsafe_allow_html=True)
    elif st.session_state.qa_chain is None:
        st.error("RAG system not initialized properly. Please check the logs.")
    else:
        with st.spinner("Analyzing SEC filing and generating answer..."):
            try:
                # Get response
                result = st.session_state.qa_chain(question)
                
                # Display answer
                st.markdown("<div class='answer-box'>", unsafe_allow_html=True)
                st.markdown(f"**Answer:** {result['result']}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display sources
                st.markdown("### Sources / Citations")
                for i, doc in enumerate(result["source_documents"]):
                    source = doc.metadata.get("source", "Unknown")
                    content = doc.page_content
                    
                    with st.expander(f"Source {i+1}: {os.path.basename(source)}"):
                        st.markdown(f"<div class='source-content'>{content}</div>", unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Make sure:\n1. The Chroma database has been created by running ingest.py\n2. Ollama is running with the llama3 model\n3. All required packages are installed")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>SEC Filing Summarizer & Q&A | Powered by RAG</div>", unsafe_allow_html=True)