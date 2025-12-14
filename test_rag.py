import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("Testing RAG components...")

# Check if database exists
DB_PATH = "chroma_db"
if not os.path.exists(DB_PATH):
    print("❌ Chroma database not found")
    exit(1)

print("✅ Chroma database found")

# Load embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embeddings loaded")
except Exception as e:
    print("❌ Failed to load embeddings:", str(e))
    exit(1)

# Load vector store
try:
    vectorstore = Chroma(persist_directory=DB_PATH, embedding=embeddings)
    print("✅ Vector store loaded")
except Exception as e:
    print("❌ Failed to load vector store:", str(e))
    exit(1)

# Create retriever
try:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ Retriever created")
except Exception as e:
    print("❌ Failed to create retriever:", str(e))
    exit(1)

# Test LLM connection
try:
    llm = OllamaLLM(model="mistral")
    test_response = llm.invoke("Hello")
    print("✅ LLM connected successfully")
except Exception as e:
    print("❌ Failed to connect to LLM:", str(e))
    exit(1)

# Create prompt template
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
prompt = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
try:
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain created successfully")
except Exception as e:
    print("❌ Failed to create RAG chain:", str(e))
    exit(1)

# Test the chain
try:
    result = qa_chain.invoke("What is the company's revenue?")
    print("✅ RAG chain working correctly")
    print("Sample response:", result[:100] + "...")
except Exception as e:
    print("❌ Failed to test RAG chain:", str(e))
    exit(1)

print("🎉 All RAG components are working correctly!")