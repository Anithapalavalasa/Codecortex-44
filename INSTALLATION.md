# Installation Guide

## Prerequisites

Before installing the required packages, you need to have:

1. Python 3.8 or higher
2. Microsoft Visual C++ Build Tools (for Windows) or equivalent build tools for your OS
3. Rust compiler (for chromadb)

## Option 1: Install with Conda (Recommended)

Using conda can help avoid many of the build issues:

```bash
# Create a new conda environment
conda create -n sec-rag python=3.9

# Activate the environment
conda activate sec-rag

# Install packages
conda install -c conda-forge langchain chromadb sentence-transformers
pip install streamlit ollama langchain-community
```

## Option 2: Install with Pip (Standard Approach)

If you prefer to use pip, you'll need to install the build tools first:

### For Windows:

1. Install Microsoft C++ Build Tools:
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "C++ build tools" workload

2. Install Rust:
   - Download from: https://www.rust-lang.org/tools/install
   - Or run in PowerShell: `winget install Rustlang.Rustup`

3. Install packages:
   ```bash
   pip install langchain langchain-community langchain-huggingface langchain-chroma langchain-ollama chromadb sentence-transformers streamlit ollama
   ```

## Option 3: Use Pre-compiled Wheels

Try installing with pre-compiled wheels to avoid building from source:

```bash
pip install --only-binary=all langchain langchain-community langchain-huggingface langchain-chroma langchain-ollama chromadb sentence-transformers streamlit ollama
```

## Troubleshooting

### If you encounter issues with chromadb:

1. Try installing a specific version:
   ```bash
   pip install chromadb==0.4.22
   ```

2. Or install with conda:
   ```bash
   conda install -c conda-forge chromadb
   ```

### If you encounter issues with sentence-transformers:

1. Install PyTorch first:
   ```bash
   pip install torch
   pip install sentence-transformers
   ```

## Running the Application

Once you have the packages installed:

1. Run the ingestion script (first time only):
   ```bash
   python ingest.py
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## Installing Ollama (for local LLM)

1. Download Ollama from: https://ollama.com/download
2. Install and run Ollama
3. Pull the llama3 model:
   ```bash
   ollama pull llama3
   ```