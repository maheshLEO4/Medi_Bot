
# 🩺 Medi_Bot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![RAG](https://img.shields.io/badge/Powered%20By-RAG-orange)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Groq%20%7C%20Local-lightblue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)

---

## 🧭 What is Medi_Bot?

**Medi_Bot** is an **AI-powered medical assistant** built using **Retrieval-Augmented Generation (RAG)** and a **Streamlit web interface**.  
It combines **vector-based document retrieval** with a **language model (LLM)** to provide accurate, context-aware medical answers — all through an intuitive chat interface.

Think of it as a **virtual AI doctor’s assistant** — capable of reading your medical data, retrieving relevant context from stored medical literature, and generating intelligent summaries and explanations.

---

## 🎯 What Does Medi_Bot Do?

- Reads and understands medical or clinical documents stored in the `data/` folder  
- Uses **embeddings** to convert text into numerical vectors  
- Stores and retrieves this data efficiently using a **vector database (FAISS / Qdrant)**  
- Accepts user queries from a **Streamlit web interface**  
- Retrieves the most relevant medical passages  
- Passes this retrieved context to an **LLM** (OpenAI, Groq, or local model)  
- Generates concise, factual, and contextually accurate **medical responses**  

---

## 💡 Why is Medi_Bot Useful?

Medi_Bot is built for:
- 🧑‍⚕️ **Medical students** — to quickly find summarized answers from medical books  
- 🧬 **Healthcare professionals** — for quick reference or contextual decision support  
- 🧑‍💻 **AI enthusiasts** — to learn how RAG-based chatbots function end-to-end  
- 🏥 **Hospitals & institutions** — to create domain-specific medical assistants  

**Key Benefits:**
- 🔍 Accurate, fact-grounded, and context-aware answers  
- 📚 Works on your **own medical data** — fully local and private  
- ⚡ Fast and efficient vector-based retrieval  
- 💬 User-friendly **Streamlit chat interface**  
- 🧠 (Optional) Memory feature to retain conversation context  

---

## 🚀 Features

- Streamlit-based **interactive chat UI**  
- Local or cloud **vector database** (FAISS / Qdrant)  
- Context-aware **RAG pipeline** for medical Q&A  
- Optional **persistent memory**  
- Privacy-friendly: all your data stays local if desired  

---

## 📂 Project Structure

| File / Folder | Description |
|----------------|-------------|
| `app.py` | Main Streamlit app file — runs the chatbot interface |
| `build_db.py` | Builds the vector database from medical documents in `data/` |
| `create_memory_for_llm.py` | Initializes or updates long-term memory for the LLM |
| `query_processing.py` | Handles user query logic, retrieval, and response generation |
| `vector_store.py` | Manages vector storage, retrieval, and persistence |
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

### 4️⃣ (Optional) Initialize LLM memory

```powershell
python create_memory_for_llm.py
```

### 5️⃣ Run the Streamlit app

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`) to start chatting with **Medi_Bot**.

---

## 🧠 How Medi_Bot Works

1. **Ingest:** `build_db.py` reads medical documents and creates embeddings.
2. **Store:** Embeddings are stored in a vector database (`vector_store.py`).
3. **Query:** User asks a question via the Streamlit chat.
4. **Retrieve:** The system fetches top-matching passages.
5. **Generate:** The LLM uses these passages to craft an answer.
6. **Memory (optional):** Context from previous conversations can be stored for continuity.

---

## 💬 Usage Modes

* **Streamlit Chat (default):**
  Interactive web interface — type your question and get AI-powered answers.

* **API Mode (optional):**
  Can be extended to serve endpoints like `/query` via Flask or FastAPI.

---

## 🔧 Configuration

### Environment Variables


| Variable         | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `QDRANT_URL`     | URL endpoint of your Qdrant vector database.                 |
| `QDRANT_API_KEY` | API key used to authenticate access to your Qdrant instance. |
| `GROQ_API_KEY`   | API key for accessing the Groq LLM API.                      |
| `OPENAI_API_KEY` | Required if using OpenAI embeddings or LLMs.                 |



---

## 🧩 Troubleshooting

| Issue                                         | Solution                                               |
| --------------------------------------------- | ------------------------------------------------------ |
| `Import 'vector_store' could not be resolved` | Ensure the file name and imports match exactly         |
| No results returned                           | Run `build_db.py` to populate your database            |
| Long input errors                             | Use text chunking or reduce prompt size                |
| App doesn’t launch properly                   | Run using `streamlit run app.py` from the project root |

```powershell
$env:PYTHONPATH = "$env:PYTHONPATH;$(Resolve-Path .)"
```

---

## 🛠️ Development Tips

* Maintain consistent file naming and imports
* Add `__init__.py` if converting to a Python package
* Keep `requirements.txt` updated when adding new dependencies
* Use `.env` for all private configurations

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request with a detailed description

---

## 📜 License

This project is open-source under the **MIT License**.
Feel free to modify and use it for personal, educational, or research purposes.

---

## 📫 Contact

For suggestions, improvements, or bug reports — open an **issue** in this repository.

---

## 🧩 Example Workflow Summary

```text
User Query → Query Processing → Vector Retrieval → Context Selection → LLM Response → Streamlit UI
```

Medi_Bot uses a **Retrieval-Augmented Generation (RAG)** pipeline:

* Queries are matched with the most relevant stored passages
* The selected context is passed to the LLM
* The LLM generates a **factual, medically-grounded response**

---

### ❤️ Built with Python, Streamlit, FAISS, Qdrant & OpenAI

