import streamlit as st
import os
import pandas as pd
import time
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

# Custom CSS for SEC Filing Summarizer & Q&A (RAG) look and feel
st.markdown("""
<style>
/* Light background for better visibility */
body {
    background: #f8f9fa;
    color: #212529;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Card styling with light background */
.glass-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #dee2e6;
    transition: all 0.3s ease;
}

.glass-card:hover {
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.12);
}

/* Header styling */
.header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    border: 1px solid #dee2e6;
}

.title {
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #1a73e8;
}

.subtitle {
    font-size: 1.3rem;
    color: #495057;
    margin-bottom: 1rem;
}

/* Document selector */
.document-selector {
    background: #ffffff;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #dee2e6;
}

.document-tag {
    display: inline-block;
    background: #e3f2fd;
    color: #1a73e8;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    border: 1px solid #bbdefb;
}

/* Input styling - HIGH VISIBILITY */
.stTextInput > div > div > input {
    background-color: #ffffff !important;
    border: 2px solid #1a73e8 !important;
    border-radius: 8px !important;
    color: #212529 !important;
    padding: 1rem !important;
    font-size: 1.1rem !important;
    box-shadow: 0 2px 4px rgba(26, 115, 232, 0.1);
}

.stTextInput > div > div > input:focus {
    border-color: #0d47a1 !important;
    box-shadow: 0 0 0 3px rgba(26, 115, 232, 0.2) !important;
}

/* Button styling - HIGH VISIBILITY */
.stButton > button {
    background: #ffffff !important;
    color: #1a73e8 !important;
    border: 2px solid #1a73e8 !important;
    border-radius: 8px !important;
    padding: 1rem 1.5rem !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 8px rgba(26, 115, 232, 0.2) !important;
}

.stButton > button:hover {
    background: #e3f2fd !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(26, 115, 232, 0.3) !important;
    color: #0d47a1 !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 4px rgba(26, 115, 232, 0.2) !important;
}

/* Answer box styling - HIGH VISIBILITY */
.answer-card {
    background: #ffffff;
    border-left: 5px solid #1a73e8;
    padding: 1.8rem;
    border-radius: 0 12px 12px 0;
    margin-top: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    border: 1px solid #dee2e6;
    color: #212529;
}

.confidence-indicator {
    height: 10px;
    background: #e9ecef;
    border-radius: 5px;
    margin: 15px 0;
    position: relative;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #e53935, #fbc02d, #43a047);
    border-radius: 5px;
}

.confidence-label {
    text-align: right;
    font-size: 0.9rem;
    color: #6c757d;
    margin-top: 5px;
}

.source-card {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
    color: #212529;
}

.source-header {
    font-weight: 600;
    color: #1a73e8;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #dee2e6;
}

.source-content {
    margin-top: 1rem;
    font-size: 0.95rem;
    color: #212529;
    line-height: 1.6;
}

.section-tag {
    display: inline-block;
    background: #e3f2fd;
    color: #1a73e8;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.5rem;
    border: 1px solid #bbdefb;
}

/* Comparison view */
.comparison-container {
    display: flex;
    gap: 20px;
    margin-top: 2rem;
}

.comparison-card {
    flex: 1;
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid #dee2e6;
}

/* Loading spinner */
.stSpinner > div {
    color: #1a73e8 !important;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.9rem;
    color: #6c757d;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #dee2e6;
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

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Comparison", "💬 Chat", "📈 Charts"])

with tab1:
    # Header
    st.markdown("""
    <div class="header">
        <h1 class="title">🏛️ SEC Filing Summarizer & Q&A (RAG)</h1>
        <p class="subtitle">Analyze SEC filings with AI-powered insights and问答</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard metrics
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Filings", value="24")
    with col2:
        st.metric(label="Companies", value="8")
    with col3:
        st.metric(label="Questions Asked", value="127")
    with col4:
        st.metric(label="Avg. Response", value="1.2s")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Filings per Year")
        
        import pandas as pd
        import numpy as np
        
        # Sample data
        years = ['2018', '2019', '2020', '2021', '2022', '2023']
        filings = [3, 4, 5, 6, 7, 8]
        
        chart_data = pd.DataFrame({'Year': years, 'Filings': filings})
        st.bar_chart(chart_data.set_index('Year'))
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Questions per Company")
        
        # Sample data
        companies = ['Apple', 'Microsoft', 'Amazon', 'Google', 'Tesla', 'Meta']
        questions = [25, 22, 18, 20, 15, 12]
        
        chart_data = pd.DataFrame({'Company': companies, 'Questions': questions})
        st.bar_chart(chart_data.set_index('Company'))
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Comparison view
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Document Comparison")
    
    # Initialize session state variables if not exists
    if 'comparison_question' not in st.session_state:
        st.session_state.comparison_question = ""
    if 'comparison_answers' not in st.session_state:
        st.session_state.comparison_answers = {}
    if 'selected_documents' not in st.session_state:
        st.session_state.selected_documents = []
    
    # Comparison controls
    col1, col2 = st.columns([3, 1])
    with col1:
        comparison_question = st.text_input("Enter question for comparison:", st.session_state.comparison_question)
    with col2:
        st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
        compare_clicked = st.button("🔍 Compare")
        st.markdown("</div>", unsafe_allow_html=True)
    
    if compare_clicked and comparison_question:
        st.session_state.comparison_question = comparison_question
        
        # Simulate comparison results
        st.session_state.comparison_answers = {
            "Apple Inc. (10-K 2023)": f"Apple's response to '{comparison_question}': Strong performance with 5.2% revenue growth and strategic investments in AR/VR technologies.",
            "Microsoft Corp. (10-K 2023)": f"Microsoft's response to '{comparison_question}': Focused on cloud expansion with Azure driving 12.8% revenue growth and enterprise adoption."
        }
    
    # Display comparison results
    if st.session_state.comparison_answers:
        st.markdown(f"<h4>Comparison Results for: {st.session_state.comparison_question}</h4>", unsafe_allow_html=True)
        
        comparison_cols = st.columns(len(st.session_state.comparison_answers))
        for i, (doc, answer) in enumerate(st.session_state.comparison_answers.items()):
            with comparison_cols[i]:
                st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
                st.markdown(f"#### 📄 {doc}")
                st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.comparison_question:
        st.info("Enter a question and click 'Compare' to see results.")
    else:
        st.info("👈 Enter a question to compare across selected documents.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    # Chat view (existing content)
    # Document selection
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Add Documents")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        company = st.selectbox("Select Company (Tab)", ["Apple Inc.", "Microsoft Corp.", "Amazon.com Inc.", "Google LLC", "Tesla Inc.", "Meta Platforms Inc.", "NVIDIA Corp.", "Netflix Inc."], key="company_tab3")
    with col2:
        doc_type = st.selectbox("Document Type (Tab)", ["10-K", "10-Q", "8-K", "S-1"], key="doc_type_tab3")
    with col3:
        year = st.selectbox("Filing Year (Tab)", ["2023", "2022", "2021", "2020", "2019", "2018"], key="year_tab3")
    
    if st.button("➕ Add Document", key="add_doc"):
        # Initialize selected_documents if not exists
        if 'selected_documents' not in st.session_state:
            st.session_state.selected_documents = []
            
        if company:
            doc_id = f"{company} - {doc_type} ({year})"
            if doc_id not in st.session_state.selected_documents:
                st.session_state.selected_documents.append(doc_id)
                st.success("Document added!")
            else:
                st.warning("Already added!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Selected documents
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Selected Documents")
    
    # Initialize selected_documents if not exists
    if 'selected_documents' not in st.session_state:
        st.session_state.selected_documents = []
        
    if st.session_state.selected_documents:
        st.markdown(f"<p>Documents selected: {len(st.session_state.selected_documents)}/10</p>", unsafe_allow_html=True)
        
        for i, doc in enumerate(st.session_state.selected_documents):
            col1, col2 = st.columns([9, 1])
            with col1:
                st.markdown(f'<div class="document-tag">{doc}</div>', unsafe_allow_html=True)
            with col2:
                if st.button("❌", key=f"remove_tab3_{i}"):
                    st.session_state.selected_documents.remove(doc)
                    st.rerun()
    else:
        st.info("Add documents above to start.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat interface
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Chat with SEC Filings")
    
    # Display messages
    # Initialize messages if not exists
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "answer" in message:
                st.markdown(f'<div class="answer-card">{message["answer"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask about SEC filings..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate response generation
            response = "This is a simulated response. In a real implementation, this would connect to your RAG system."
            
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "",
            "answer": full_response
        })
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    # Charts view
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Detailed Analytics")
    
    # Sample line chart
    st.markdown("#### 📅 Quarterly Trends")
    quarters = ['Q1 2022', 'Q2 2022', 'Q3 2022', 'Q4 2022', 'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023']
    values = [25, 30, 28, 35, 40, 45, 42, 50]
    
    chart_data = pd.DataFrame({'Quarter': quarters, 'Activity': values})
    st.line_chart(chart_data.set_index('Quarter'))
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Section coverage chart
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📚 Section Coverage")
    
    sections = ['Risk Factors', 'MD&A', 'Financial Statements', 'Legal Proceedings']
    coverage = [95, 87, 92, 78]
    
    chart_data = pd.DataFrame({'Section': sections, 'Coverage %': coverage})
    st.bar_chart(chart_data.set_index('Section'))
    
    st.markdown('</div>', unsafe_allow_html=True)

# Initialize session state
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'comparison_question' not in st.session_state:
    st.session_state.comparison_question = ""
if 'comparison_answers' not in st.session_state:
    st.session_state.comparison_answers = {}
if 'company' not in st.session_state:
    st.session_state.company = ""
if 'doc_type' not in st.session_state:
    st.session_state.doc_type = "10-K"
if 'year' not in st.session_state:
    st.session_state.year = "2023"

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
        
        st.info("Loading embeddings and vector store...")
        
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
                st.info("Attempting to connect to Ollama with mistral model...")
                llm = Ollama(model="mistral")
                test_response = llm.invoke("Hello")
                st.success("Connected to Ollama with mistral model")
            except Exception as e1:
                st.warning(f"Failed to connect to mistral: {e1}")
                try:
                    st.info("Attempting to connect to Ollama with llama3 model...")
                    llm = Ollama(model="llama3")
                    test_response = llm.invoke("Hello")
                    st.success("Connected to Ollama with llama3 model")
                except Exception as e2:
                    st.error(f"Failed to connect to llama3: {e2}")
                    raise e2
        except Exception as e:
            # Create a simple mock LLM for demonstration
            st.warning(f"Failed to connect to Ollama: {e}. Using mock responses.")
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
        
        st.success("RAG components initialized successfully!")
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
    doc_type = st.selectbox("Document Type", ["10-K", "10-Q", "8-K", "S-1"], key="doc_type_selector_main")
with col2:
    year = st.selectbox("Filing Year", ["2023", "2022", "2021", "2020", "2019", "2018"], key="year_selector_main")
with col3:
    company = st.selectbox("Select Company", ["Apple Inc.", "Microsoft Corp.", "Amazon.com Inc.", "Google LLC", "Tesla Inc."], index=0, key="company_selector_main")

# Add document button
if st.button("➕ Add to List"):
    doc_identifier = f"{company} - {doc_type} ({year})"
    # Initialize selected_documents if not exists
    if 'selected_documents' not in st.session_state:
        st.session_state.selected_documents = []
        
    if doc_identifier not in st.session_state.selected_documents:
        st.session_state.selected_documents.append(doc_identifier)
        st.success(f"Added {doc_identifier} to your document list!")
    else:
        st.warning(f"{doc_identifier} is already in your document list!")

# Display selected documents
# Initialize selected_documents if not exists
if 'selected_documents' not in st.session_state:
    st.session_state.selected_documents = []
    
if st.session_state.selected_documents:
    st.markdown("### 📁 Selected Documents:")
    for i, doc in enumerate(st.session_state.selected_documents):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f'<span class="document-tag">{doc}</span>', unsafe_allow_html=True)
        with col2:
            if st.button("❌", key=f"remove_main_{i}"):
                st.session_state.selected_documents.remove(doc)
                st.rerun()
else:
    st.info("👆 Use the document selector above to start adding documents. Add up to 10 documents to start your conversation.")

st.markdown('</div>', unsafe_allow_html=True)

# Chat Interface
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 💬 Chat with SEC Filing Summarizer & Q&A (RAG)")

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
        
        # Initialize selected_documents if not exists
        if 'selected_documents' not in st.session_state:
            st.session_state.selected_documents = []
            
        if not st.session_state.selected_documents:
            full_response = "Please select documents first using the document selector above. Add up to 10 documents to start your conversation with the SEC Filing Summarizer & Q&A (RAG)."
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