import os
import sys

print("Python executable:", sys.executable)
print("Python path:", sys.path)

# Try different import approaches
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ langchain_huggingface import successful")
except ImportError as e:
    print("❌ langchain_huggingface import failed:", str(e))
    
    # Try alternative import
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ langchain_community.embeddings import successful")
    except ImportError as e2:
        print("❌ langchain_community.embeddings import failed:", str(e2))

# Test other imports
try:
    from langchain_chroma import Chroma
    print("✅ langchain_chroma import successful")
except ImportError as e:
    print("❌ langchain_chroma import failed:", str(e))

try:
    from langchain_ollama import OllamaLLM
    print("✅ langchain_ollama import successful")
except ImportError as e:
    print("❌ langchain_ollama import failed:", str(e))