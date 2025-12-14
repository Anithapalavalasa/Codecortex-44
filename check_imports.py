try:
    import pandas
    print("pandas ok")
    import requests
    print("requests ok")
    import bs4
    print("bs4 ok")
    import pypdf
    print("pypdf ok")
    import langchain_community
    print("langchain_community ok")
    import chromadb
    print("chromadb ok")
    import sentence_transformers
    print("sentence_transformers ok")
except Exception as e:
    print(f"Import Error: {e}")
