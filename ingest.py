from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

DATA_PATH = "data/sec_filings.csv"
DB_PATH = "chroma_db"
DEMO_MODE = True  # Use a small subset for demo

# Safety check
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File not found: {DATA_PATH}")

print("📄 Loading SEC filings from CSV...")
loader = CSVLoader(file_path=DATA_PATH, encoding="utf-8")
documents = loader.load()

# Demo subset for faster testing
if DEMO_MODE:
    documents = documents[:100]

print(f"✅ Loaded {len(documents)} rows")

# Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"🧩 Created {len(chunks)} chunks")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_PATH
)

vectorstore.persist()
print("🎉 CSV ingestion complete. Data stored in ChromaDB.")
print("💡 Set DEMO_MODE = False to process the full dataset")
