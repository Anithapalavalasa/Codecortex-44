import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Page configuration
st.set_page_config(
    page_title="SEC Filing Explorer",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .subheader {
        font-size: 1.5rem;
        color: #306998;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .answer-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #ddd;
    }
    .source-box {
        background-color: #e8f4f8;
        border-left: 5px solid #4B8BBE;
        padding: 15px;
        margin-top: 15px;
        border-radius: 0 5px 5px 0;
        font-size: 0.9rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #666;
        font-size: 0.9rem;
        padding: 20px;
        border-top: 1px solid #eee;
    }
    .warning-box {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 5px 5px 0;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 5px 5px 0;
    }
    .success-box {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        margin: 15px 0;
        border-radius: 0 5px 5px 0;
    }
    .stButton>button {
        border-radius: 20px;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .subheader {
            font-size: 1.2rem;
        }
        .answer-box, .source-box {
            padding: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>🔍 SEC Filing Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Ask questions about SEC filings and get AI-powered answers with sources</p>", unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.qa_chain = None

@st.cache_resource
def initialize_rag():
    """Initialize the RAG components"""
    try:
        DB_PATH = "chroma_db"
        
        # Check if database exists
        if not os.path.exists(DB_PATH):
            st.error("Chroma database not found. Please run the ingestion script first.")
            return None, None
        
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

# Main UI
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2 class='subheader'>Ask a Question</h2>", unsafe_allow_html=True)
    
    # Example questions
    example_questions = [
        "What are the company's main business segments?",
        "What are the key financial highlights?",
        "What are the major risks identified by the company?",
        "Who are the key executives and their compensation?",
        "What is the company's strategy for growth?"
    ]
    
    # Question input
    question = st.text_input("Enter your question about the SEC filing:", placeholder="e.g., What are the company's main revenue streams?")
    
    # Example buttons
    st.markdown("<p><small>Try these examples:</small></p>", unsafe_allow_html=True)
    example_cols = st.columns(len(example_questions))
    
    for i, (col, example) in enumerate(zip(example_cols, example_questions)):
        if col.button(example, key=f"example_{i}", help="Click to use this example question"):
            question = example
    
    # Submit button
    submit_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)

with col2:
    st.markdown("<h2 class='subheader'>About This Tool</h2>", unsafe_allow_html=True)
    st.markdown('''<div class="info-box"><strong>This tool uses Retrieval-Augmented Generation (RAG) to answer questions about SEC filings.</strong><br><br>
    <strong>How it works:</strong>
    <ol>
        <li>Your question is embedded and matched against relevant sections of the SEC filing</li>
        <li>The most relevant passages are retrieved</li>
        <li>An AI language model generates an answer based on these passages</li>
    </ol>
    <strong>Note:</strong> Make sure Ollama is running with the llama3 model for best results.
    </div>''', unsafe_allow_html=True)

# Process question
if submit_button and question:
    if st.session_state.qa_chain is None:
        st.error("RAG system not initialized properly. Please check the logs.")
    else:
        with st.spinner("Analyzing SEC filing and generating answer..."):
            try:
                # Get response
                result = st.session_state.qa_chain(question)
                
                # Display answer
                st.markdown("<h2 class='subheader'>Answer</h2>", unsafe_allow_html=True)
                st.markdown(f"<div class='answer-box'><p>{result['result']}</p></div>", unsafe_allow_html=True)
                
                # Show success indicator
                st.markdown('''<div class="success-box">✅ Answer generated successfully with sources below</div>''', unsafe_allow_html=True)
                
                # Display sources
                st.markdown("<h3 class='subheader'>Sources</h3>", unsafe_allow_html=True)
                for i, doc in enumerate(result["source_documents"]):
                    source = doc.metadata.get("source", "Unknown")
                    content = doc.page_content
                    
                    with st.expander(f"Source {i+1}: {os.path.basename(source)}"):
                        st.markdown(f"<div class='source-box'>{content}</div>", unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.markdown('''<div class="warning-box">⚠️ Please check the following:<br>
                1. The Chroma database has been created by running ingest.py<br>
                2. Ollama is running with the llama3 model<br>
                3. All required packages are installed<br>
                4. The data file exists in the data/ directory</div>''', unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>SEC Filing Explorer | Powered by LangChain, ChromaDB, and Ollama</div>", unsafe_allow_html=True)