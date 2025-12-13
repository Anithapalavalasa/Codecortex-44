#!/usr/bin/env python3
"""
Test script to verify that all required packages can be imported.
"""

import sys

def test_import(package_name):
    try:
        __import__(package_name)
        print(f"✓ {package_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ {package_name} import failed: {e}")
        return False

def main():
    print("Testing package imports...")
    print("=" * 40)
    
    packages = [
        "streamlit",
        "langchain",
        "langchain_community",
        "chromadb",
        "sentence_transformers",
        "ollama"
    ]
    
    results = []
    for package in packages:
        results.append(test_import(package))
    
    print("=" * 40)
    if all(results):
        print("All packages imported successfully!")
        return 0
    else:
        print("Some packages failed to import. Please check INSTALLATION.md for help.")
        return 1

if __name__ == "__main__":
    sys.exit(main())