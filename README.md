# 🏦 SEC Filing Summarizer & Q&A using RAG

A Retrieval-Augmented Generation (RAG) system that enables natural language question-answering on SEC filings (10-K, 10-Q) with verifiable source citations.

## 📋 Problem Statement

Investors, analysts, and students face significant difficulty in quickly understanding large and complex SEC filings. These documents are:
- **Lengthy**: Often 100+ pages
- **Unstructured**: Dense financial and legal text
- **Domain-specific**: Require expertise to extract insights

**Objective**: Build a Generative AI–powered system that allows users to:
- ✅ Ask natural language questions about SEC filings
- ✅ Receive accurate, context-aware answers
- ✅ View verifiable citations from the original document

This system reduces analysis time, improves decision-making, and demonstrates real-world financial document intelligence.

## 🎯 Domain

- **Primary**: Finance
- **Secondary**: Productivity, Education

## 🏗️ Solution Overview

We propose a **Retrieval-Augmented Generation (RAG)** based system that:
1. Processes real SEC filings (PDF/TXT)
2. Chunks documents into semantically meaningful sections
3. Generates embeddings for each chunk
4. Stores embeddings in ChromaDB vector database
5. Retrieves relevant chunks based on user queries
6. Uses an LLM to generate grounded answers with citations

### High-Level Architecture

```
SEC Filing (PDF)
    ↓
Document Parser (PyPDF)
    ↓
Text Chunking (RecursiveCharacterTextSplitter)
    ↓
Embedding Generation (sentence-transformers)
    ↓
Vector Database (ChromaDB)
    ↓
Query → Retrieve Relevant Chunks
    ↓
LLM (LLaMA 3) → Generate Answer + Citations
```

## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **LLM** | LLaMA 3 (via Ollama – Local LLM) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB |
| **Framework** | LangChain |
| **Document Parsing** | PyPDF, unstructured |

## 📊 Data Source

**SEC Filings Dataset (Kaggle)**: https://www.kaggle.com/datasets/kharanshuvalangar/sec-filings

**Sample used in demo**: Apple Inc. 10-K filing

## 📁 Project Structure

```
sec-rag/
│
├── data/
│   └── apple_10k.pdf          # SEC filing (add your own)
│
├── ingest.py                   # Document loading, chunking, embeddings
├── rag.py                      # Question answering with citations
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── chroma_db/                  # Vector store (auto-created)
```

## 🚀 Setup & Run Instructions

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Local LLM (Ollama)

```bash
# Install Ollama from https://ollama.ai

# Pull LLaMA 3 model
ollama pull llama3

# Verify installation
ollama run llama3 "Hello, world!"
```

**Note**: Ensure Ollama is running (`ollama serve`) before using the RAG system.

### Step 3: Prepare SEC Filing

1. Download a SEC filing (10-K, 10-Q) from [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch.html) or use the Kaggle dataset
2. Place the PDF file in the `data/` directory
3. Example: `data/apple_10k.pdf`

### Step 4: Ingest SEC Filing

```bash
python ingest.py
```

Or specify a custom file path:
```bash
python ingest.py data/your_filing.pdf
```

This will:
- Load and parse the PDF
- Chunk the document into semantic sections
- Generate embeddings
- Store in ChromaDB

**Expected output**:
```
Loading document: apple_10k.pdf
Loaded 150 pages/sections
Chunking documents...
Created 450 chunks
Generating embeddings and storing in vector database...
✅ Successfully ingested 450 chunks into 'sec_filings' collection
```

### Step 5: Ask Questions

**Interactive Mode**:
```bash
python rag.py
```

**Single Question**:
```bash
python rag.py "What are the major risk factors mentioned?"
```

## 💡 Example Queries

- "What are the major risk factors mentioned?"
- "Summarize management discussion and analysis"
- "Are there any litigation risks?"
- "How did revenue change year over year?"
- "What is the company's business model?"
- "What are the key financial metrics?"
- "Describe the competitive landscape"

## 📝 Example Output

```
🔍 Question: What are the major risk factors mentioned?

📝 Answer:
Based on the SEC filing, the major risk factors include:

1. **Market Competition**: The company faces intense competition in the technology sector...
2. **Supply Chain Risks**: Dependencies on third-party manufacturers and suppliers...
3. **Regulatory Changes**: Potential impact of new regulations on operations...
4. **Cybersecurity Threats**: Risk of data breaches and cyber attacks...

[Source citations included]

📚 Sources (4 chunks):
--------------------------------------------------------------------------------

[1] Source: apple_10k | Page: 12
    Snippet: Item 1A. Risk Factors. The Company's business, financial condition...

[2] Source: apple_10k | Page: 13
    Snippet: Competition. The markets for the Company's products and services...
```

## ✅ Evaluation & Guardrails

### Evaluation Approach

1. **Manual Factual Verification**: Answers verified against original document using citations
2. **Consistency Check**: Repeated queries tested for consistency
3. **Retrieval Relevance**: Top-k chunk accuracy measured
4. **Citation Accuracy**: Source references validated

### Guardrails

- ✅ **Context-only answers**: System restricted to retrieved context only
- ✅ **Mandatory citations**: Every response includes source citations
- ✅ **No hallucination**: Financial numbers not generated without source
- ✅ **Explicit uncertainty**: System states when answer cannot be found in context

### Limitations

- Performance depends on chunking quality
- Financial interpretation is informational, not advisory
- Local LLM may have slower response times than cloud APIs
- Large documents may require more computational resources

## 🎯 Innovation & Impact

### Innovation

- ✅ Combines real financial documents with RAG architecture
- ✅ Provides explainable AI via mandatory citations
- ✅ Uses local LLM (Ollama) to reduce dependency on paid APIs
- ✅ Semantic chunking for better context retrieval

### Impact & Expandability

This system can be extended to:

- **Multi-company comparison**: Compare risk factors across companies
- **Risk scoring agents**: Automated risk assessment
- **Financial trend analysis**: Time-series analysis across filings
- **Web interface**: FastAPI or Streamlit UI
- **Real-time updates**: Process new filings automatically
- **Multi-document RAG**: Query across multiple filings simultaneously

## 🔧 Configuration

### Customize Embedding Model

Edit `ingest.py` and `rag.py`:
```python
embedding_model = "sentence-transformers/all-mpnet-base-v2"  # Larger, more accurate
```

### Adjust Chunk Size

Edit `ingest.py`:
```python
chunk_size=1500,      # Larger chunks (more context)
chunk_overlap=300     # More overlap
```

### Change LLM Model

Edit `rag.py`:
```python
llm_model = "llama3:8b"  # or "mistral", "codellama", etc.
```

### Retrieve More Context

Edit `rag.py`:
```python
k = 6  # Retrieve top 6 chunks instead of 4
```

## 🐛 Troubleshooting

### Ollama Connection Error

```bash
# Ensure Ollama is running
ollama serve

# Verify model is installed
ollama list

# Pull model if missing
ollama pull llama3
```

### Vector Database Not Found

```bash
# Run ingestion first
python ingest.py
```

### Memory Issues

- Reduce `chunk_size` in `ingest.py`
- Reduce `k` (retrieval count) in `rag.py`
- Use smaller embedding model

## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama Documentation](https://github.com/ollama/ollama)
- [SEC EDGAR Database](https://www.sec.gov/edgar/searchedgar/companysearch.html)

## 👤 Author

**Palavalasa Anitha**  
B.Tech IT, JNTU-GV  
GenAI Hackathon Participant

## 📄 License

This project is created for educational and hackathon purposes.

---

**Note**: This system is for informational purposes only and does not constitute financial advice. Always consult with qualified financial professionals for investment decisions.

