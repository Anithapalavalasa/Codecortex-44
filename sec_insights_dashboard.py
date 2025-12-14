import streamlit as st
import pandas as pd
import numpy as np
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
    page_title="SEC Filing Summarizer & Q&A (RAG)",
    page_icon="🏛️",
    layout="wide"
)

# Custom CSS for modern dashboard design
st.markdown("""
<style>
/* Modern dark gradient background */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
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

/* Card styling with glassmorphism effect */
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

/* Metric styling */
.metric-card {
    background: rgba(15, 23, 42, 0.8);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(94, 129, 172, 0.3);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #3498db;
    margin: 10px 0;
}

.metric-label {
    font-size: 1rem;
    color: #94a3b8;
}

/* Document chip */
.document-chip {
    display: inline-flex;
    align-items: center;
    background: rgba(52, 152, 219, 0.2);
    color: #3498db;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 500;
    margin: 4px;
    border: 1px solid rgba(52, 152, 219, 0.3);
}

.document-chip .remove-btn {
    margin-left: 8px;
    cursor: pointer;
    color: #ff6b6b;
    font-weight: bold;
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

.stSelectbox > div > div {
    background-color: rgba(30, 41, 59, 0.7) !important;
    border: 1px solid rgba(94, 129, 172, 0.5) !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    padding: 0.5rem !important;
    backdrop-filter: blur(5px);
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

/* Chat styling */
.chat-message {
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    max-width: 80%;
}

.user-message {
    background: rgba(41, 128, 185, 0.2);
    border-left: 4px solid #3498db;
    margin-left: auto;
}

.assistant-message {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(94, 129, 172, 0.3);
}

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

.source-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(94, 129, 172, 0.3);
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
    backdrop-filter: blur(8px);
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
    st.session_state.current_view = 'dashboard'  # dashboard, comparison, chat
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

# Header
st.markdown("""
<div class="header">
    <h1 class="title">🏛️ SEC Filing Summarizer & Q&A</h1>
    <p class="subtitle">Empowering Business Intelligence with AI-Powered SEC Analysis</p>
</div>
""", unsafe_allow_html=True)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Comparison", "💬 Chat"])

# Dashboard Tab
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Key Metrics")
    
    # Create metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Filings</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">24</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Companies</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">8</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Questions Asked</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">127</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Model Used</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">Mistral</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg. Response</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">2.4s</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Filings per Year")
        
        # Sample data for filings per year
        years = ['2018', '2019', '2020', '2021', '2022', '2023']
        filings = [3, 4, 5, 6, 7, 8]
        
        fig = px.bar(x=years, y=filings, 
                     labels={'x': 'Year', 'y': 'Number of Filings'},
                     color=filings, 
                     color_continuous_scale='Blues')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Question Count per Company")
        
        # Sample data for questions per company
        companies = ['Apple', 'Microsoft', 'Amazon', 'Google', 'Tesla', 'Meta']
        questions = [25, 22, 18, 20, 15, 12]
        
        fig = px.pie(values=questions, names=companies, 
                     color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e0e0e0')
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent queries section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🕒 Recent Queries")
    
    # Sample recent queries
    recent_queries = [
        {"query": "What are the main business risks for Apple?", "time": "2 mins ago"},
        {"query": "Compare revenue growth between Microsoft and Google", "time": "15 mins ago"},
        {"query": "What is Tesla's strategy for battery production?", "time": "1 hour ago"},
        {"query": "Show debt-to-equity ratios for all companies", "time": "3 hours ago"}
    ]
    
    for query in recent_queries:
        st.markdown(f"""
        <div style="padding: 10px; margin: 5px 0; border-left: 3px solid #3498db; background: rgba(30, 41, 59, 0.4);">
            <div>{query['query']}</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">{query['time']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Comparison Tab
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Document Comparison")
    
    # Document selection for comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Select Documents for Comparison")
        if st.session_state.selected_documents:
            selected_for_comparison = st.multiselect(
                "Choose documents to compare:",
                st.session_state.selected_documents,
                default=st.session_state.selected_documents[:2] if len(st.session_state.selected_documents) >= 2 else st.session_state.selected_documents
            )
        else:
            st.info("No documents selected. Add documents in the chat interface first.")
            selected_for_comparison = []
    
    with col2:
        st.markdown("#### Comparison Question")
        comparison_question = st.text_input("Enter a question to compare across documents:")
        
        if st.button("🔍 Compare Answers") and comparison_question and len(selected_for_comparison) >= 2:
            st.session_state.comparison_question = comparison_question
            st.session_state.comparison_answers = {}
            
            # Simulate getting answers for each document
            for doc in selected_for_comparison:
                # In a real implementation, this would query the RAG system for each document
                answer = f"This is a simulated answer for the question '{comparison_question}' based on the {doc} filing. In a real implementation, this would retrieve specific information from that document."
                st.session_state.comparison_answers[doc] = answer
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display comparison results
    if st.session_state.comparison_answers:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"### 📋 Comparison Results: {st.session_state.comparison_question}")
        
        # Create comparison cards
        comparison_cols = st.columns(len(st.session_state.comparison_answers))
        
        for i, (doc, answer) in enumerate(st.session_state.comparison_answers.items()):
            with comparison_cols[i]:
                st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
                st.markdown(f"#### 📄 {doc}")
                st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Highlight differences (simulated)
        st.markdown("### ⚡ Key Differences")
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.7); border-left: 5px solid #e74c3c; padding: 15px; border-radius: 0 10px 10px 0;">
            <ul>
                <li><strong>Revenue Growth:</strong> Apple (12.5%) vs Microsoft (15.2%)</li>
                <li><strong>R&D Investment:</strong> Apple ($2.3B) vs Microsoft ($3.1B)</li>
                <li><strong>Market Strategy:</strong> Apple focuses on premium hardware, Microsoft on cloud services</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Chat Tab
with tab3:
    # Document selection section
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Selected documents panel
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
    
    # Chat interface
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
                st.markdown('<div class="confidence-indicator"><div class="confidence-fill" style="width: {}%"></div></div>'.format(message["confidence"]), unsafe_allow_html=True)
                st.markdown(f'<div class="confidence-label">Confidence: {message["confidence"]}%</div>', unsafe_allow_html=True)
            
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

# Section-wise document coverage chart
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📚 Section-wise Document Coverage")

# Sample data for section coverage
sections = ['Risk Factors', 'Management Discussion', 'Financial Statements', 'Legal Proceedings', 'Market Risk']
coverage = [95, 87, 92, 78, 85]

fig = go.Figure(data=[go.Bar(
    x=coverage,
    y=sections,
    orientation='h',
    marker=dict(
        color=coverage,
        colorscale='Blues',
        showscale=False
    )
)])

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e0e0e0'),
    xaxis=dict(showgrid=False, range=[0, 100]),
    yaxis=dict(showgrid=False),
    height=300
)

st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">SEC Filing Summarizer & Q&A (RAG) | Powered by LangChain, ChromaDB, and Ollama</div>', unsafe_allow_html=True)