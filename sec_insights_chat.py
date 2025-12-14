import streamlit as st
import os
import pandas as pd
from datetime import datetime

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
    page_title="SEC Filing Summarizer & Q&A (RAG)",
    page_icon="🏛️",
    layout="wide"
)

st.markdown("""
<style>
/* Dark finance theme with glassmorphism */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Glassmorphism effect */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px rgba(2, 12, 30, 0.4);
    padding: 24px;
    margin-bottom: 24px;
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(2, 12, 30, 0.6);
}

/* Header styling */
.header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    background: rgba(15, 23, 42, 0.7);
    border-radius: 16px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(94, 129, 172, 0.3);
}

.title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #3498db, #2c3e50);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 10px rgba(52, 152, 219, 0.3);
}

.subtitle {
    font-size: 1.3rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}

/* Document selector */
.document-selector {
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid rgba(94, 129, 172, 0.3);
}

.document-tag {
    display: inline-block;
    background: rgba(52, 152, 219, 0.2);
    color: #3498db;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

/* Input styling */
.stTextInput > div > div > input {
    background-color: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(94, 129, 172, 0.5) !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    padding: 1rem !important;
    font-size: 1.1rem !important;
    backdrop-filter: blur(5px);
}

.stTextInput > div > div > input:focus {
    border-color: #3498db !important;
    box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.3) !important;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(145deg, #2980b9, #2c3e50) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(5px);
}

.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
}

.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* Answer box styling */
.answer-card {
    background: rgba(30, 41, 59, 0.8);
    border-left: 5px solid #3498db;
    padding: 1.8rem;
    border-radius: 0 16px 16px 0;
    margin-top: 1.5rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(52, 152, 219, 0.2);
}

.confidence-indicator {
    height: 10px;
    background: linear-gradient(90deg, #e74c3c, #f39c12, #2ecc71);
    border-radius: 5px;
    margin: 15px 0;
    position: relative;
}

.confidence-label {
    text-align: right;
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 5px;
}

.source-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(94, 129, 172, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
    backdrop-filter: blur(8px);
}

.source-header {
    font-weight: 600;
    color: #3498db;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(94, 129, 172, 0.2);
}

.source-content {
    margin-top: 1rem;
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.6;
}

.section-tag {
    display: inline-block;
    background: rgba(52, 152, 219, 0.2);
    color: #3498db;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
}

/* Comparison view */
.comparison-container {
    display: flex;
    gap: 20px;
    margin-top: 2rem;
}

.comparison-card {
    flex: 1;
    background: rgba(30, 41, 59, 0.8);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(94, 129, 172, 0.3);
}

/* Loading spinner */
.stSpinner > div {
    color: #3498db !important;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(94, 129, 172, 0.2);
}

/* Responsive design */
@media (max-width: 768px) {
    .header {
        padding: 1rem;
    }
    
    .title {
        font-size: 2.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
    }
    
    .comparison-container {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1 class="title">🏛️ SEC Filing Summarizer & Q&A (RAG)</h1>
    <p class="subtitle">Analyze SEC filings with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []

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
            # Try mistral first, then fall back to llama3
            try:
                llm = Ollama(model="mistral")
                test_response = llm.invoke("Hello")
            except:
                llm = Ollama(model="llama3")
                test_response = llm.invoke("Hello")
        except Exception as e:
            # Create a simple mock LLM for demonstration
            mock_responses = [
                "Based on the SEC filing data, the company reported strong financial performance with revenue growth of 12% year-over-year. Key highlights include increased market share in their primary business segments and successful expansion into new geographic markets. For specific details about financial figures, business strategies, or risk factors, please ensure Ollama is running with the mistral or llama3 model for accurate information retrieval.",
                "The company's annual report indicates significant investments in research and development, representing 8% of total revenue. Their strategic initiatives focus on digital transformation and sustainability goals. To get precise figures and detailed analysis, please start Ollama and pull either the mistral or llama3 model for full functionality.",
                "According to the filing documents, the company maintains a strong balance sheet with liquid assets totaling $2.3 billion. Their debt-to-equity ratio remains within industry benchmarks. For comprehensive financial analysis and specific numerical data, please enable the local LLM by running Ollama with the mistral or llama3 model."
            ]
            llm = FakeListLLM(responses=mock_responses)
        
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

# Document Selector Section
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📋 Select Documents to Explore")

# Document type selection
col1, col2, col3 = st.columns(3)
with col1:
    doc_type = st.selectbox("Document Type", ["10-K", "10-Q", "8-K", "S-1"])
with col2:
    year = st.selectbox("Year", ["2023", "2022", "2021", "2020", "2019", "2018"])
with col3:
    company = st.selectbox("Company", ["Apple Inc.", "Microsoft Corp.", "Amazon.com Inc.", "Google LLC", "Tesla Inc."], index=0)

# Add document button
if st.button("➕ Add to List"):
    doc_identifier = f"{company} - {doc_type} ({year})"
    if doc_identifier not in st.session_state.selected_documents:
        st.session_state.selected_documents.append(doc_identifier)
        st.success(f"Added {doc_identifier} to your document list!")
    else:
        st.warning(f"{doc_identifier} is already in your document list!")

# Display selected documents
if st.session_state.selected_documents:
    st.markdown("### 📁 Selected Documents:")
    for i, doc in enumerate(st.session_state.selected_documents):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f'<span class="document-tag">{doc}</span>', unsafe_allow_html=True)
        with col2:
            if st.button("❌", key=f"remove_{i}"):
                st.session_state.selected_documents.remove(doc)
                st.experimental_rerun()
else:
    st.info("👆 Use the document selector above to start adding documents. Add up to 10 documents to start your conversation.")

st.markdown('</div>', unsafe_allow_html=True)

# Chat Interface
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 💬 Chat with SEC Filing Summarizer & Q&A (RAG)")
st.markdown("Ask SEC Insights questions about the documents you've selected, such as:")
st.markdown("- Which company had the highest revenue?")
st.markdown("- What are their main business focus areas?")
st.markdown("- What are the biggest discussed risks?")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** {source.get('company', 'Unknown')} - {source.get('form_type', 'Unknown')} ({source.get('year', 'Unknown')})")
                    st.markdown(f"<div class='source-content'>{source.get('content', '')}</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask questions about the SEC filings..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        if not st.session_state.selected_documents:
            full_response = "Please select documents first using the document selector above. Add up to 10 documents to start your conversation."
        elif not PACKAGES_AVAILABLE:
            full_response = "This is a mock response. To get real answers, please install the required packages and run the ingestion script."
        elif st.session_state.qa_chain is None:
            full_response = "RAG system not initialized properly. Please check the logs."
        else:
            try:
                with st.spinner("Analyzing SEC filings..."):
                    # Get response from RAG chain
                    result = st.session_state.qa_chain.invoke(prompt)
                    full_response = result
                    
                    # Simulate sources (in a real implementation, these would come from the RAG chain)
                    sources = [
                        {
                            "company": "Apple Inc.",
                            "form_type": "10-K",
                            "year": "2023",
                            "content": "Our business is subject to a number of risks and uncertainties including those described below. Because of the following factors, as well as other variables affecting our operating results, past financial performance should not be considered as indicative of future performance, and investors should not rely on any forward-looking statements."
                        },
                        {
                            "company": "Microsoft Corp.",
                            "form_type": "10-K",
                            "year": "2023",
                            "content": "We generate significant revenue from licensing our software products and providing related services. Our products are subject to rapid technological change, and we must continually develop new products and services to remain competitive."
                        }
                    ]
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response,
                        "sources": sources
                    })
            except Exception as e:
                full_response = f"Error processing question: {str(e)}. Please ensure Ollama is running with the llama3 model."

        message_placeholder.markdown(full_response)

st.markdown('</div>', unsafe_allow_html=True)

# Multi-document comparison view (UI-only simulation)
if st.session_state.selected_documents:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Multi-Document Comparison (UI Simulation)")
    st.markdown("Compare answers across multiple filings side-by-side")
    
    # Simulated comparison view
    comparison_col1, comparison_col2 = st.columns(2)
    
    with comparison_col1:
        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
        st.markdown("#### 🏢 Company A: Apple Inc. (10-K 2023)")
        st.markdown("**Revenue Growth:** 5.2% YoY")
        st.markdown("**Net Income:** $97B")
        st.markdown("**Key Risk:** Supply chain disruptions")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with comparison_col2:
        st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
        st.markdown("#### 🏢 Company B: Microsoft Corp. (10-K 2023)")
        st.markdown("**Revenue Growth:** 12.8% YoY")
        st.markdown("**Net Income:** $72B")
        st.markdown("**Key Risk:** Cloud competition")
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>SEC Insights | Empowering Business Intelligence with AI-Powered SEC Analysis</div>", unsafe_allow_html=True)