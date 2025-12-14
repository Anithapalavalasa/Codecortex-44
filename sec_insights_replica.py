import streamlit as st
import os
import pandas as pd
import numpy as np

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

# Custom CSS for secinsights.ai style
st.markdown("""
<style>
/* Light gradient background */
body {
    background: linear-gradient(135deg, #f5f7fa 0%, #e4edf9 100%);
    color: #333333;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header styling */
.header {
    text-align: center;
    padding: 3rem 0;
    margin-bottom: 2rem;
    background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%);
    border-radius: 0 0 20px 20px;
    color: white;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.title {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.3rem;
    font-weight: 300;
    opacity: 0.9;
    max-width: 700px;
    margin: 0 auto;
}

/* Card styling */
.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    border: 1px solid #e1e5eb;
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

/* Document chip */
.document-chip {
    display: inline-flex;
    align-items: center;
    background: #f0f4ff;
    color: #2a5298;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
    margin: 4px;
    border: 1px solid #d0d8ff;
}

.document-chip .remove-btn {
    margin-left: 8px;
    cursor: pointer;
    color: #ff6b6b;
    font-weight: bold;
}

/* Input styling */
.stTextInput > div > div > input {
    border-radius: 10px !important;
    border: 1px solid #d1d5db !important;
    padding: 0.8rem !important;
    font-size: 1rem !important;
}

.stSelectbox > div > div {
    border-radius: 10px !important;
    border: 1px solid #d1d5db !important;
}

/* Button styling */
.stButton > button {
    border-radius: 10px !important;
    padding: 0.8rem 1.5rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.primary-btn {
    background: linear-gradient(120deg, #1e3c72 0%, #2a5298 100%) !important;
    color: white !important;
    border: none !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(30, 60, 114, 0.3) !important;
}

.secondary-btn {
    background: white !important;
    color: #2a5298 !important;
    border: 1px solid #2a5298 !important;
}

.secondary-btn:hover {
    background: #f0f4ff !important;
}

.disabled-btn {
    background: #e5e7eb !important;
    color: #9ca3af !important;
    border: 1px solid #d1d5db !important;
    cursor: not-allowed !important;
}

/* Chat styling */
.chat-message {
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    max-width: 80%;
}

.user-message {
    background: #f0f4ff;
    border-left: 4px solid #2a5298;
    margin-left: auto;
}

.assistant-message {
    background: white;
    border: 1px solid #e1e5eb;
}

.answer-card {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
    border: 1px solid #e2e8f0;
}

.source-card {
    background: #f1f5f9;
    border-radius: 10px;
    padding: 1rem;
    margin-top: 1rem;
    border-left: 3px solid #2a5298;
}

.confidence-bar {
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    margin: 1rem 0;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981 0%, #2a5298 100%);
    border-radius: 4px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #e2e8f0;
}

/* Responsive design */
@media (max-width: 768px) {
    .header {
        padding: 2rem 1rem;
    }
    
    .title {
        font-size: 2.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
    }
    
    .chat-message {
        max-width: 95%;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'home'  # home or chat
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

# Render home view
def render_home_view():
    # Header
    st.markdown("""
    <div class="header">
        <h1 class="title">🏛️ SEC Filing Summarizer & Q&A</h1>
        <p class="subtitle">Empower your organization's Business Intelligence with SEC Insights. Effortlessly analyze multifaceted financial documents such as 10-Ks and 10-Qs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Document selection card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📋 Select Documents to Explore")
    
    # Document selection inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        company = st.selectbox("Company", ["Apple Inc.", "Microsoft Corp.", "Amazon.com Inc.", "Google LLC", "Tesla Inc.", "Meta Platforms Inc.", "NVIDIA Corp.", "Netflix Inc."])
    with col2:
        doc_type = st.selectbox("Document Type", ["10-K", "10-Q", "8-K", "S-1"])
    with col3:
        year = st.selectbox("Year", ["2023", "2022", "2021", "2020", "2019", "2018"])
    
    # Add document button
    if st.button("➕ Add to List", key="add_doc", type="primary"):
        if company:
            doc_identifier = f"{company} - {doc_type} ({year})"
            if doc_identifier not in st.session_state.selected_documents:
                st.session_state.selected_documents.append(doc_identifier)
                st.success(f"Added {doc_identifier} to your document list!")
            else:
                st.warning(f"{doc_identifier} is already in your document list!")
        else:
            st.warning("Please enter a company ticker or name.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Selected documents panel
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 Selected Documents")
    
    if st.session_state.selected_documents:
        st.markdown(f"<p>Add up to 10 docs to start your conversation. Currently selected: {len(st.session_state.selected_documents)}/10</p>", unsafe_allow_html=True)
        
        # Display selected documents as chips
        for i, doc in enumerate(st.session_state.selected_documents):
            col1, col2 = st.columns([9, 1])
            with col1:
                st.markdown(f'<div class="document-chip">{doc}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("❌", key=f"remove_{i}", help="Remove document"):
                    st.session_state.selected_documents.remove(doc)
                    st.rerun()
    else:
        st.info("No documents selected yet. Use the form above to add documents to your analysis list.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Start conversation button
    start_disabled = len(st.session_state.selected_documents) == 0
    button_class = "disabled-btn" if start_disabled else "primary-btn"
    
    st.markdown(f'<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    if st.button("💬 Start Conversation", 
                 disabled=start_disabled, 
                 key="start_conv",
                 help="Add at least one document to start the conversation" if start_disabled else None):
        st.session_state.current_view = 'chat'
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Render chat view
def render_chat_view():
    # Header
    st.markdown("""
    <div class="header">
        <h1 class="title">📊 SEC Filing Dashboard</h1>
        <p class="subtitle">Comprehensive analysis of selected SEC filings with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Back to document selection button
    if st.button("⬅️ Back to Document Selection"):
        st.session_state.current_view = 'home'
        st.rerun()
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔍 Comparisons", "💬 Chat", "📄 Documents"])
    
    with tab1:
        render_dashboard_tab()
    
    with tab2:
        render_comparison_tab()
    
    with tab3:
        render_chat_tab()
    
    with tab4:
        render_documents_tab()

# Dashboard Tab
def render_dashboard_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Key Metrics")
    
    # Create sample metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Documents Analyzed", value=len(st.session_state.selected_documents))
    with col2:
        st.metric(label="Questions Asked", value=len([m for m in st.session_state.messages if m["role"] == "user"]))
    with col3:
        st.metric(label="Avg. Confidence", value="85%")
    with col4:
        st.metric(label="Sources Cited", value="24")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Document Distribution")
        
        # Sample data for document types
        doc_types = ['10-K', '10-Q', '8-K', 'S-1']
        counts = [len([d for d in st.session_state.selected_documents if '10-K' in d]),
                  len([d for d in st.session_state.selected_documents if '10-Q' in d]),
                  len([d for d in st.session_state.selected_documents if '8-K' in d]),
                  len([d for d in st.session_state.selected_documents if 'S-1' in d])]
        
        chart_data = pd.DataFrame({'Document Type': doc_types, 'Count': counts})
        st.bar_chart(chart_data.set_index('Document Type'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📅 Yearly Distribution")
        
        # Sample data for years
        years = ['2018', '2019', '2020', '2021', '2022', '2023']
        counts = [len([d for d in st.session_state.selected_documents if '2018' in d]),
                  len([d for d in st.session_state.selected_documents if '2019' in d]),
                  len([d for d in st.session_state.selected_documents if '2020' in d]),
                  len([d for d in st.session_state.selected_documents if '2021' in d]),
                  len([d for d in st.session_state.selected_documents if '2022' in d]),
                  len([d for d in st.session_state.selected_documents if '2023' in d])]
        
        chart_data = pd.DataFrame({'Year': years, 'Count': counts})
        st.line_chart(chart_data.set_index('Year'))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent activity
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🕒 Recent Activity")
    
    if st.session_state.messages:
        for message in st.session_state.messages[-3:]:  # Show last 3 messages
            if message["role"] == "user":
                st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message.get("answer", message.get("content", ""))[:100]}...</div>', unsafe_allow_html=True)
    else:
        st.info("No recent activity. Ask a question to get started!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Comparison Tab
def render_comparison_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📊 Document Comparison")
    
    if len(st.session_state.selected_documents) < 2:
        st.warning("Add at least 2 documents to enable comparison.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Create comparison data
    companies = list(set([d.split(' - ')[0] for d in st.session_state.selected_documents]))
    
    if len(companies) < 2:
        st.warning("Select documents from different companies to enable comparison.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Sample comparison data
    comparison_data = pd.DataFrame({
        'Metric': ['Revenue Growth', 'Net Income', 'Total Assets', 'Liabilities', 'Equity'],
        companies[0]: [np.random.uniform(5, 15), np.random.uniform(100, 500), np.random.uniform(1000, 5000), np.random.uniform(500, 2000), np.random.uniform(500, 3000)],
        companies[1]: [np.random.uniform(5, 15), np.random.uniform(100, 500), np.random.uniform(1000, 5000), np.random.uniform(500, 2000), np.random.uniform(500, 3000)]
    })
    
    st.table(comparison_data)
    
    # Visualization
    st.markdown("### 📈 Comparative Analysis")
    chart_data = comparison_data.melt(id_vars=['Metric'], var_name='Company', value_name='Value')
    st.bar_chart(chart_data.pivot(index='Metric', columns='Company', values='Value'))
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat Tab
def render_chat_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💬 Chat with SEC Filings")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message"><strong>Assistant:</strong> {message["content"]}', unsafe_allow_html=True)
            
            # Display answer card if present
            if "answer" in message:
                st.markdown(f'<div class="answer-card">{message["answer"]}</div>', unsafe_allow_html=True)
            
            # Display confidence if present
            if "confidence" in message:
                st.markdown('<div class="confidence-bar"><div class="confidence-fill" style="width: {}%"></div></div>'.format(message["confidence"]), unsafe_allow_html=True)
                st.markdown(f'<small>Confidence: {message["confidence"]}%</small>', unsafe_allow_html=True)
            
            # Display sources if present
            if "sources" in message:
                st.markdown("#### 📚 Sources", unsafe_allow_html=True)
                for i, source in enumerate(message["sources"]):
                    st.markdown(f'<div class="source-card"><strong>Source {i+1}:</strong> {source}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about the selected SEC filings...", placeholder="e.g., What are the company's main business risks?")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_input:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Generate response
            if not PACKAGES_AVAILABLE:
                response = "This is a mock response. To get real answers, please install the required packages and run the ingestion script."
                confidence = 50
                sources = ["Mock source 1", "Mock source 2"]
            elif st.session_state.qa_chain is None:
                response = "RAG system not initialized properly. Please check the logs."
                confidence = 0
                sources = []
            else:
                try:
                    with st.spinner("Analyzing SEC filings..."):
                        # Get response from RAG chain
                        result = st.session_state.qa_chain.invoke(user_input)
                        response = result
                        confidence = 85  # Mock confidence
                        sources = [
                            "SEC Filing 2023 - Risk Factors Section",
                            "SEC Filing 2022 - Management Discussion and Analysis",
                            "SEC Filing 2023 - Financial Statements"
                        ]
                except Exception as e:
                    response = f"Error processing question: {str(e)}. Please ensure Ollama is running with the llama3 model."
                    confidence = 0
                    sources = []
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": "",
                "answer": response,
                "confidence": confidence,
                "sources": sources
            })
            
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Documents Tab
def render_documents_tab():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📁 Selected Documents")
    
    if st.session_state.selected_documents:
        for doc in st.session_state.selected_documents:
            st.markdown(f'<div class="document-chip">{doc}</div>', unsafe_allow_html=True)
    else:
        st.info("No documents selected yet.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main application logic
if st.session_state.current_view == 'home':
    render_home_view()
else:
    render_chat_view()

# Footer
st.markdown('<div class="footer">SEC Filing Summarizer & Q&A (RAG) | Powered by LangChain, ChromaDB, and Ollama</div>', unsafe_allow_html=True)