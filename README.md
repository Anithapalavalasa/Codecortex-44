SEC Filing Summarizer & Q&A using RAG
1. Problem Statement

SEC filings such as 10-K and 10-Q are critical for investors, analysts, and students, but they are:

Extremely long (100+ pages)

Unstructured and dense

Difficult to analyze without financial expertise

Extracting insights like risk factors, management discussion, litigation risks, and financial performance is time-consuming and error-prone.

Objective:
Build a Generative AI–powered system that enables users to:

Ask natural language questions about SEC filings

Receive accurate, context-aware answers

View verifiable citations from the original document

This system reduces analysis time and demonstrates real-world Financial Document Intelligence using RAG.

2. Domain

Primary: Finance 🏦

Secondary: Productivity, Education

3. Solution Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline to answer questions from real SEC filings.

High-Level Workflow

Load SEC filing (PDF / TXT)

Chunk document into semantic sections

Generate embeddings for each chunk

Store embeddings in ChromaDB

Retrieve relevant chunks based on user query

Use an LLM to generate grounded answers

Display answers with source citations

4. Technologies Used

Language: Python

LLM: LLaMA 3 (via Ollama – Local LLM)

Embeddings: sentence-transformers (MiniLM)

Vector Database: ChromaDB

Framework: LangChain

Document Parsing: unstructured, PyPDF

Optional APIs: Groq / OpenAI (configurable)

5. Data Source

Dataset: SEC Filings Dataset (Kaggle)

Link: https://www.kaggle.com/datasets/kharanshuvalangar/sec-filings

Sample Used in Demo:

Apple Inc. – 10-K Filing

6. Project Structure
sec-rag/
│
├── data/
│   └── apple_10k.pdf
│
├── ingest.py        # Document loading, chunking, embeddings
├── rag.py           # Question answering with citations
├── requirements.txt
├── README.md
└── chroma_db/       # Vector store (auto-created)
7. Setup & Run Instructions
Step 1: Environment Setup
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate       # Windows
pip install -r requirements.txt
Step 2: Prepare Local LLM
ollama pull llama3
ollama run llama3
Step 3: Ingest SEC Filing
python ingest.py

This will:

Load the SEC filing

Chunk the document

Generate embeddings

Store them in ChromaDB

Step 4: Ask Questions
python rag.py
8. Example Queries

"What are the major risk factors mentioned?"

"Summarize management discussion"

"Are there any litigation risks?"

"How did revenue change year over year?"

9. Evaluation & Guardrails
Evaluation Approach

Manual factual verification using citations

Consistency checks across repeated queries

Retrieval relevance (Top-k chunk accuracy)

Guardrails

Answers strictly limited to retrieved context

Mandatory source citation for every response

No hallucinated financial numbers

Informational use only (not financial advice)

10. Limitations

Performance depends on chunking strategy

Complex financial interpretation may require human judgment

Currently supports single-document querying

11. Innovation & Impact
Innovation

Real-world financial document RAG system

Explainable AI through citations

Fully local LLM (no paid API dependency)

Impact & Expandability

Can be extended to:

Multi-company comparison

Risk scoring agents

Financial trend analysis

FastAPI / Streamlit interface

12. Hackathon Compliance Checklist




13. Demo Video

(To be added – YouTube link)

14. Author

Palavalasa Anitha
B.Tech IT, JNTU-GV
GenAI Hackathon Participant
