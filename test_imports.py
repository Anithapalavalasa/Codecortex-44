import sys
import os

print("Testing imports...")

# Test langchain_huggingface
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ langchain_huggingface: OK")
except ImportError as e:
    print("❌ langchain_huggingface: FAILED -", str(e))
    
# Test langchain_chroma
try:
    from langchain_chroma import Chroma
    print("✅ langchain_chroma: OK")
except ImportError as e:
    print("❌ langchain_chroma: FAILED -", str(e))

# Test langchain_ollama
try:
    from langchain_ollama import OllamaLLM
    print("✅ langchain_ollama: OK")
except ImportError as e:
    print("❌ langchain_ollama: FAILED -", str(e))

# Test basic langchain components
try:
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    print("✅ langchain_core components: OK")
except ImportError as e:
    print("❌ langchain_core components: FAILED -", str(e))

# Check if database exists
if os.path.exists("chroma_db"):
    print("✅ Chroma database: FOUND")
else:
    print("❌ Chroma database: NOT FOUND")

print("Import test completed.")