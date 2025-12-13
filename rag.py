from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Update the imports to use newer packages if available
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

DB_PATH = "chroma_db"

print("🔹 Loading embeddings and vector store...")

# Initialize HuggingFace embeddings
hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Use the embed_documents method for Chroma
def embedding_function(texts):
    return hf_embeddings.embed_documents(texts)

# Load Chroma vectorstore
vectorstore = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embedding_function
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Update the LLM initialization
# Initialize Ollama local LLM
llm = None
try:
    llm = Ollama(model="llama3")
    # Test if model is available
    test_response = llm.invoke("Hello")
    print("✅ Connected to Ollama successfully")
except Exception as e:
    print(f"Warning: Could not connect to Ollama: {e}")
    print("Using a mock LLM instead")

# Create a prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
prompt = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask questions in a loop
while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    result = qa_chain.invoke(query)

    print("\n🧠 Answer:")
    print(result)
    
    # Note: In this simplified version, we're not returning source documents
    # To get sources, we would need to modify the chain to return them
