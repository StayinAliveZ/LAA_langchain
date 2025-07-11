# 🔬 学术洞察引擎 (Academic Insight Engine)

This project is a Streamlit web application that uses a Retrieval-Augmented Generation (RAG) pipeline with LangChain to analyze and answer questions about academic papers in PDF format. It leverages large language models to provide insightful, context-aware answers based on the content of the uploaded documents.

## ✨ Features

- **PDF Analysis**: Upload one or more academic papers for analysis.
- **Vector Database**: Uses FAISS to create a searchable vector store from the document content.
- **Conversational AI**: Ask questions in natural language about the papers.
- **Advanced Chunking**: Employs `unstructured`'s title-based chunking for more logical text segmentation.
- **Clean UI**: A simple and intuitive user interface built with Streamlit.

## 🛠️ Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.8+
- Git

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

*(Replace `YOUR_USERNAME` and `YOUR_REPOSITORY_NAME` after you create the GitHub repo).*

### 3. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install all the required packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

The application requires API keys for the language model and embeddings.

1. Create a new file named `.env` in the root of the project directory.
2. Add your API keys to the `.env` file as follows:

   ```env
   DEEPSEEK_API_KEY="your_deepseek_api_key"
   dashscope_api_key="your_dashscope_api_key"
   ```

   * **DEEPSEEK_API_KEY**: Get this from the DeepSeek platform.
   * **dashscope_api_key**: Get this from Alibaba Cloud's DashScope platform.

## 🚀 How to Run

Once the setup is complete, you can run the Streamlit application with the following command:

```bash
streamlit run langchain_rag.py
```

Your web browser should automatically open to the application's URL (usually `http://localhost:8501`).

## 📖 How to Use

1. **Upload Documents**: Use the sidebar to upload one or more PDF files.
2. **Build Knowledge Base**: Click the "🚀 构建知识库" (Build Knowledge Base) button. The app will process the PDFs, chunk the content, and store it in a FAISS vector database.
3. **Ask Questions**: Once the knowledge base is ready, type your questions about the documents in the main input box and press Enter.
4. **Get Answers**: The AI assistant will analyze the documents and provide a detailed answer based on the context.
5. **Clear Knowledge Base**: If you want to analyze new documents, you can clear the existing database using the "🗑️ 清除知识库" (Clear Knowledge Base) button.
