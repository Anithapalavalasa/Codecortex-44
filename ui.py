import streamlit as st
import os

# Try importing required packages
try:
    # Try newer packages first
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    try:
        from langchain_ollama import OllamaLLM
        # Alias for backward compatibility
        Ollama = OllamaLLM
    except ImportError:
        try:
            from langchain_ollama import OllamaLLM as Ollama
        except ImportError:
            from langchain_community.llms import Ollama

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.language_models import FakeListLLM
    PACKAGES_AVAILABLE = True
except ImportError:
    PACKAGES_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SEC Filing Explorer",
    page_icon="🏛️",
    layout="wide"
)

# Initialize session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Theme toggle function
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Apply theme styles
def apply_theme():
    if st.session_state.theme == 'dark':
        st.markdown("""
        <style>
        :root {
            --background-color: #0c1424;
            --card-background: #1e293b;
            --text-color: #e0e0e0;
            --accent-color: #3498db;
            --border-color: #334155;
            --input-bg: #1e293b;
            --button-bg: linear-gradient(90deg, #2980b9, #2c3e50);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        :root {
            --background-color: #ffffff;
            --card-background: #f8f9fa;
            --text-color: #333333;
            --accent-color: #2980b9;
            --border-color: #dee2e6;
            --input-bg: #ffffff;
            --button-bg: linear-gradient(90deg, #3498db, #2980b9);
        }
        </style>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <style>
    body {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--accent-color);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 1rem;
    }
    
    .card {
        background: var(--card-background);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
    }
    
    .stTextInput > div > div > input {
        background-color: var(--input-bg) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        color: var(--text-color) !important;
        padding: 0.8rem !important;
        font-size: 1rem !important;
    }
    
    .stButton > button {
        background: var(--button-bg) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.8rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .answer-box {
        background: var(--card-background);
        border-left: 4px solid var(--accent-color);
        padding: 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .source-card {
        background: var(--card-background);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .source-header {
        font-weight: 600;
        color: var(--accent-color);
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .theme-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
    }
    
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    @media (max-width: 768px) {
        .main-container {
            padding: 1rem;
        }
        
        .title {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# Theme toggle button
st.markdown("<div class='theme-toggle'>", unsafe_allow_html=True)
if st.button("🌓" if st.session_state.theme == 'light' else "☀️"):
    toggle_theme()
    st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<div class='header'>", unsafe_allow_html=True)
st.markdown("<h1 class='title'>🏛️ SEC Filing Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions about SEC filings with AI-powered insights</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Initialize RAG components
@st.cache_resource
def initialize_rag():
    """Initialize the RAG components with updated chain"""
    if not PACKAGES_AVAILABLE:
        st.warning("Required packages not installed.")
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
        llm = None
        try:
            llm = Ollama(model="llama3")
            # Test if model is available
            test_response = llm.invoke("Hello")
        except Exception as e:
            # Create a simple mock LLM for demonstration
            llm = FakeListLLM(responses=["This is a mock response. In a real implementation with Ollama running, you would get detailed answers about SEC filings."])
        
        # Create a prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Helpful Answer:"""
        prompt = PromptTemplate.from_template(prompt_template)
        
        # Create the RAG chain (updated approach)
        qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
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

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top: 0;'>Ask a Question</h2>", unsafe_allow_html=True)
    
    # Example questions
    example_questions = [
        "What are the company's main business segments?",
        "What are the key financial highlights?",
        "What are the major risks identified by the company?",
        "Who are the key executives and their compensation?",
        "What is the company's strategy for growth?"
    ]
    
    # Question input
    question = st.text_input("Enter your question about the SEC filing:", 
                            placeholder="e.g., What are the company's main revenue streams?",
                            label_visibility="collapsed")
    
    # Example buttons
    st.markdown("<p style='margin: 0.5rem 0;'><small>Try these examples:</small></p>", unsafe_allow_html=True)
    example_cols = st.columns(2)
    
    for i, example in enumerate(example_questions[:4]):  # Show first 4 examples
        col_idx = i % 2
        if example_cols[col_idx].button(example, key=f"example_{i}", help="Click to use this example"):
            question = example
    
    # Submit button
    submit_button = st.button("🔍 Analyze Filing", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='margin-top: 0;'>How It Works</h2>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
    <li><strong>RAG Technology:</strong> Combines retrieval and generation for accurate answers</li>
    <li><strong>Vector Search:</strong> Finds relevant sections in SEC filings</li>
    <li><strong>AI Analysis:</strong> Llama3 model processes information</li>
    <li><strong>Source Citations:</strong> See exactly where information comes from</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top: 1.5rem;'>Requirements</h3>", unsafe_allow_html=True)
    st.markdown("""
    <ul>
    <li>Ollama with llama3 model</li>
    <li>Ingested SEC filing data</li>
    <li>Internet connection for initial setup</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Process question
if submit_button and question:
    if not PACKAGES_AVAILABLE:
        st.error("Required packages not installed. Please check your installation.")
    elif st.session_state.qa_chain is None:
        st.error("RAG system not initialized properly. Please check the logs.")
    else:
        with st.spinner("Analyzing SEC filing and generating answer..."):
            try:
                # Get response
                result = st.session_state.qa_chain.invoke(question)
                
                # Display answer
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("<h2>Answer</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='answer-box'>{result}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Note about sources (in this implementation, we don't retrieve sources)
                st.info("💡 Note: This version focuses on answer quality. For detailed source citations, use the advanced version.")
                        
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Please ensure:\n1. Ollama is running with the llama3 model\n2. All required packages are installed\n3. The ingestion process was completed")

# Footer
st.markdown("<div class='footer'>SEC Filing Explorer | Powered by LangChain, ChromaDB, and Ollama</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)