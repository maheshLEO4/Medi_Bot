
# 🩺 Medi_Bot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![RAG](https://img.shields.io/badge/Powered%20By-RAG-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Local-lightblue)

---

## 🧭 What is Medi_Bot?

**Medi_Bot** is an **AI-powered medical assistant** built using **Retrieval-Augmented Generation (RAG)**.  
It combines **vector-based document retrieval** with a **language model (LLM)** to provide accurate, context-aware medical information.

You can think of it as a **local AI doctor’s assistant** — capable of reading your medical data, searching for relevant context from stored medical texts, and giving intelligent, summarized answers.

---

## 🎯 What Does Medi_Bot Do?

- Reads and understands medical or clinical documents stored in the `data/` folder  
- Uses embeddings to **convert text into numerical representations** (vectors)  
- Stores and retrieves this data efficiently using a **vector database (FAISS/Annoy)**  
- Accepts user queries and retrieves the **most relevant medical passages**  
- Passes this retrieved context to an **LLM** (like OpenAI GPT or a local model)  
- Generates clear, concise, and contextually accurate **medical answers**  

---

## 💡 Why is Medi_Bot Useful?

Medi_Bot is designed for:
- 🧑‍⚕️ **Medical students** — to quickly access summarized answers from medical textbooks or research papers  
- 🧬 **Healthcare professionals** — for quick reference or decision support  
- 🧑‍💻 **Developers & AI learners** — to understand how RAG-based assistants work  
- 🏥 **Institutions & hospitals** — to build custom, domain-specific medical chatbots with private data  

Key Benefits:
- 🔍 **Accurate and context-aware answers**
- 📚 **Uses your own medical documents** — fully local and private  
- ⚡ **Fast and efficient retrieval** using vector databases  
- 🧠 **Memory-enabled assistant** — can remember past conversations  

---

## 🚀 Features

- Build a **local vector database** from medical documents  
- Retrieve relevant context for any medical query using **embeddings**  
- Generate **intelligent, context-based responses** with an **LLM**  
- Optional **persistent memory** for long-term conversations  
- Fully **local & private** (no data leaves your system if using local models)  

---

## 📂 Project Structure

| File / Folder | Description |
|----------------|-------------|
| `app.py` | Main entry point — runs the chat or API server |
| `build_db.py` | Builds the vector database from documents in `data/` |
| `create_memory_for_llm.py` | Creates or updates persistent memory for the LLM |
| `query_processing.py` | Handles query parsing, retrieval, and orchestration between the vector store and LLM |
| `vector_store.py` | Implements vector storage, retrieval, and persistence (FAISS/Annoy, etc.) |
| `data/` | Folder containing medical source documents |
| `requirements.txt` | Python dependencies |

---

## ⚙️ Quick Start (Windows PowerShell)

### 1️⃣ Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

### 2️⃣ Install dependencies

```powershell
pip install -r requirements.txt
```

### 3️⃣ Build the vector database

```powershell
python build_db.py
```

### 4️⃣ (Optional) Initialize memory for LLM

```powershell
python create_memory_for_llm.py
```

### 5️⃣ Run the app

```powershell
python app.py
```

> 💡 If you use pipenv or conda, adjust the commands accordingly.

---

## 🧠 How Medi_Bot Works

1. **Ingest:** `build_db.py` reads medical documents and generates embeddings
2. **Store:** Embeddings are stored in a local vector database (`vector_store.py`)
3. **Query:** The user enters a medical query through CLI or API
4. **Retrieve:** The most relevant passages are fetched from the vector store
5. **Generate:** The LLM uses retrieved context to produce an accurate answer
6. **Memory (optional):** The conversation context can be stored and reused

---

## 💬 Usage Modes

* **CLI Chat:**
  Run `python app.py` and enter queries interactively.

* **API Mode:**
  If implemented, start the app to serve endpoints like `/query` (e.g., via Flask or FastAPI).

---

## 🔧 Configuration

### Environment Variables

| Variable         | Description                                 |
| ---------------- | ------------------------------------------- |
| `OPENAI_API_KEY` | Required if using OpenAI embeddings or LLMs |

### Paths

Ensure the document and database paths in `query_processing.py` and `vector_store.py` are correctly set.

---

## 🧩 Troubleshooting

| Issue                                         | Solution                                                             |
| --------------------------------------------- | -------------------------------------------------------------------- |
| `Import 'vector_store' could not be resolved` | Ensure the file is named exactly `vector_store.py` and imports match |
| No results returned                           | Run `build_db.py` to populate your database                          |
| Long input errors                             | Use chunking or reduce prompt size                                   |
| Running from another directory                | Execute scripts from the project root or set `PYTHONPATH`            |

```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$(Resolve-Path .)"
```

---

## 🛠️ Development Tips

* Maintain consistent file names and imports
* Add `__init__.py` files if converting to a Python package
* Keep `requirements.txt` updated when adding dependencies

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch for your changes
3. Submit a pull request with a clear description

---

## 📜 License

This project is open-source under the **MIT License**.
You can modify it freely for personal or educational use.

---

## 📫 Contact

For questions or feedback, open an **issue** in this repository.
You can also suggest improvements or report bugs.

---

## 🧩 Example Workflow Summary

```text
User Query → Query Processing → Vector Retrieval → Context Selection → LLM Response
```

Medi_Bot uses a **Retrieval-Augmented Generation (RAG)** approach:

* Queries are matched with the most relevant stored passages
* The selected context is passed to the LLM
* The model produces a **factual and context-aware medical answer**

---

### ❤️ Made with Python, FAISS, and OpenAI



