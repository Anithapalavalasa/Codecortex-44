# SEC Filing RAG Explorer
An AI-powered application that allows you to ask questions about SEC filings and get answers with sources using Retrieval-Augmented Generation (RAG).
<div align="center">
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)](https://streamlit.io/)
</div>
> **Note**: This project requires several Python packages that may have complex installation requirements due to native dependencies. See [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions.
## 📁 Project Structure
sec-rag/
├── app.py                 # Original Streamlit frontend application
├── streamlit_app.py       # Advanced Streamlit frontend with animations
├── ui.py                  # New basic UI with theme toggle
├── ingest.py              # Data ingestion script for processing SEC filings
├── rag.py                 # Command-line RAG implementation
├── requirements.txt       # Python dependencies
├── data/
│   └── sec_filings.csv    # SEC filing data (CSV format)
├── chroma_db/             # Vector database (created after ingestion)
└── README.md              # This file
> **Note**: The `chroma_db/` directory will be created after running the ingestion script.
## 🚀 Features
- **Natural Language Querying**: Ask questions about SEC filings in plain English
- **Source Attribution**: See exactly which parts of the filing support each answer
- **Web Interface**: User-friendly Streamlit interface
- **Local Processing**: Runs entirely on your machine with no cloud dependencies
- **Open Source Models**: Uses locally-running LLMs via Ollama

## 🛠️ Setup Instructions
### Prerequisites
1. Python 3.8 or higher
2. [Ollama](https://ollama.com/) installed and running (optional, for local LLM)
3. [Git](https://git-scm.com/) (optional, for cloning the repository)

> **Important**: Due to native dependencies in some packages (especially chromadb), you may encounter installation issues. Please refer to [INSTALLATION.md](INSTALLATION.md) for detailed installation instructions and troubleshooting tips.

### Installation
1. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd sec-rag
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Pull the required Ollama model:
   ```bash
   ollama pull mystral
   ```
> **Note**: If you encounter issues installing `chromadb` due to missing Rust dependencies, you may need to install Rust first or use pre-compiled wheels. Alternatively, you can try:
> ```bash
> pip install --only-binary=all -r requirements.txt
> ```

### Data Preparation
Ensure your SEC filing data is in CSV format and placed in the `data/sec_filings.csv` file.

The CSV should have columns appropriate for your SEC filing data. The ingestion script will process this data and create embeddings for querying.

### Ingest SEC Filings

Run the ingestion script to process the SEC filing data and create the vector database:

```bash
python ingest.py
```
This will:
- Load the SEC filing data from `data/sec_filings.csv`
- Split the text into manageable chunks
- Create embeddings using the sentence-transformers model
- Store the embeddings in a ChromaDB vector database in the `chroma_db/` directory

## ▶️ Running the Application

Common Guardrails in a RAG System (Like Yours)
1️⃣ Context-Only Answering (Anti-Hallucination)

The model is forced to answer only from retrieved SEC documents

If information is not found → AI says:

“The provided documents do not contain enough information.”

✅ Benefit: Prevents made-up financial facts

2️⃣ Source Attribution Guardrail

Every answer must include citations

If no sources are retrieved → answer is blocked

✅ Benefit: Builds trust & auditability

3️⃣ Confidence Threshold Guardrail

If similarity score from ChromaDB is low:

Do NOT generate an answer

Ask user to rephrase question

✅ Benefit: Avoids weak or misleading responses

4️⃣ Input Validation Guardrail

Blocks:

Irrelevant questions (e.g., “Who is the CEO of Google?”)

Prompt injection attempts

✅ Benefit: Protects system integrity

5️⃣ Output Format Guardrail

Answers follow a strict structure:

Answer

Sources

Confidence / Disclaimer

✅ Benefit: Professional, consistent responses

📊 Evaluations (Measuring Quality & Accuracy)

Evaluation = how well your RAG system performs.
Judges LOVE this because it shows engineering maturity.

Key RAG Evaluation Metrics (Explainable to Judges)
1️⃣ Answer Correctness

Is the answer factually correct based on the SEC filing?

Compare AI answer vs actual filing text

Manually or automatically checked

2️⃣ Groundedness (Most Important)

Is the answer strictly based on retrieved documents?

❌ Bad: AI adds extra financial advice
✅ Good: AI quotes exact SEC sections

3️⃣ Context Relevance

Are retrieved chunks actually relevant to the question?

Measured by:

Embedding similarity scores

Manual inspection

4️⃣ Faithfulness

Does the answer faithfully represent the source text without distortion?

Important for:
Risk disclosures
Revenue numbers
Legal statements

5️⃣ Latency & Performance

How fast does the system respond?

Measured by:

Retrieval time

LLM response time

6️⃣ Failure Handling

Does the system behave safely when it doesn’t know the answer?

Expected behavior:

“Information not available in the provided filings.”
### Web Interface (Recommended)

We provide multiple web interfaces for different preferences:

1. **Basic UI** (`ui.py`) - Clean, responsive interface with light/dark theme toggle
2. **Advanced UI** (`streamlit_app.py`) - Feature-rich interface with animations
3. **Simple UI** (`app.py`) - Basic interface for quick testing

To run any of the web interfaces:
```bash
streamlit run ui.py
# or
streamlit run streamlit_app.py
# or
streamlit run app.py
```

The application will open in your default web browser. You can:
- Ask questions about the SEC filing using the input field
- Try example questions with the provided buttons
- View detailed answers with source attribution
- Expand source documents to see the exact text used to generate answers
### Command-Line Interface

Alternatively, you can use the command-line interface:

```bash
python rag.py
```
This will start an interactive session where you can ask questions and receive answers in the terminal.

## 🔧 How It Works

1. **Data Ingestion**: The `ingest.py` script processes SEC filing data, breaking it into chunks and creating vector embeddings.

2. **Vector Storage**: Embeddings are stored in a ChromaDB vector database for efficient similarity search.

3. **Question Processing**: When you ask a question:
   - The question is converted to an embedding
   - Similar passages are retrieved from the vector database
   - The retrieved passages are sent to the LLM with your question
   - The LLM generates an answer based on the retrieved context

4. **Response Generation**: Answers are displayed along with the source documents that informed the response.
## 📦 Dependencies
- **LangChain**: Framework for developing applications powered by language models
- **ChromaDB**: Vector database for storing and retrieving document embeddings
- **Sentence Transformers**: State-of-the-art sentence, text, and image embeddings
- **Streamlit**: Framework for creating web applications for machine learning and data science
- **Ollama**: Tool for running large language models locally

## 🎨 UI Components
Our applications offer various user interfaces:

### Basic UI (`ui.py`)
- **Theme Toggle**: Switch between light and dark modes
- **Clean Design**: Minimalist interface focused on usability
- **Responsive Layout**: Works well on all device sizes
- **Modern Styling**: Professional appearance with smooth interactions

### Advanced UI (`streamlit_app.py`)
- **Animated Background**: Subtle floating elements for visual interest
- **Dark Finance Theme**: Professional appearance suitable for financial data
- **Enhanced Visuals**: Gradient effects and modern styling

### Simple UI (`app.py`)
- **Traditional Layout**: Classic interface with clear sections
- **Detailed Information**: Comprehensive help and about sections
- **Example Questions**: Quick-access buttons for common queries
## ⚠️ Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure Ollama is running and the llama3 model is pulled:
   ```bash
   ollama pull llama3
   ```
   You can verify the model is available by running:
   ```bash
   ollama list
   ```

2. **Database Not Found**: Run the ingestion script first:
   ```bash
   python ingest.py
   ```
   This will create the `chroma_db/` directory with the processed embeddings.

3. **Import Errors**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

4. **Runtime Errors**: If you encounter runtime errors:
   - Check that the data file exists at `data/sec_filings.csv`
   - Ensure the virtual environment is activated
   - Verify Ollama is running in the background
   - Check that the required ports are available (default: 11434 for Ollama)

5. **Performance Issues**: For better performance:
   - Close other applications while processing large datasets
   - Ensure you have at least 8GB RAM
   - The first run may take longer as models are downloaded

### Verifying Installation

To verify everything is set up correctly:

1. Check if Ollama is running:
   ```bash
   ollama --version
   ```

2. Check if required Python packages are installed:
   ```bash
   pip list | grep -E "langchain|chroma|streamlit|sentence-transformers"
   ```

### Performance Tips

- For better performance, ensure you have at least 8GB RAM
- The first run may take longer as models are downloaded
- Close other applications when processing large datasets

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
## 📞 Support


If you encounter any issues or have questions, please file an issue on the GitHub repository.


















