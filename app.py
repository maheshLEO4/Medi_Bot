import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from data_processing import get_vectorstore, set_custom_prompt, get_vectorstore_status, check_pdf_files
import os
import subprocess

if not os.path.exists("chroma_db"):  # or your DB folder
    print("📦 Building database...")
    subprocess.run(["python", "build_db.py"])


st.set_page_config(
    page_title="medi_bot - Medical Document Assistant",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 medi_bot - Medical Document Assistant")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False

# Sidebar - Status
st.sidebar.title("🔍 System Status")

status = get_vectorstore_status()
pdf_files = check_pdf_files()

st.sidebar.subheader("📊 Current Status")
st.sidebar.write(f"Data folder: {'✅' if status['data_folder_exists'] else '❌'}")
st.sidebar.write(f"PDF files: {'✅' if status['pdf_files'] else '❌'}")
st.sidebar.write(f"Chroma DB: {'✅' if status['chroma_db_exists'] else '❌'}")

if pdf_files:
    st.sidebar.subheader("📁 PDF Files")
    for pdf in pdf_files[:5]:
        st.sidebar.write(f"• {pdf}")
    if len(pdf_files) > 5:
        st.sidebar.write(f"• ... and {len(pdf_files) - 5} more")

# Load vectorstore ONCE when app starts
if not st.session_state.vectorstore_loaded:
    with st.spinner("📚 Loading MediBot knowledge base..."):
        vectorstore = get_vectorstore()
        
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.vectorstore_loaded = True
            st.success("✅ MediBot Ready!")
            
            try:
                doc_count = vectorstore._collection.count()
                st.info(f"📊 Loaded **{doc_count} knowledge chunks** from **{len(pdf_files)} medical documents**")
            except:
                st.info(f"📊 Ready with **{len(pdf_files)} medical documents**")
        else:
            st.error("""
            ❌ **Failed to load knowledge base**
            
            The Chroma database was not found or failed to load.
            This usually means the database wasn't built during deployment.
            
            **For developers:**
            Run `python build_db.py` to build the database.
            """)
            st.stop()

# Chat Interface
if st.session_state.vectorstore_loaded:
    # Header with stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Medical Documents", len(pdf_files))
    with col2:
        try:
            doc_count = st.session_state.vectorstore._collection.count()
            st.metric("Knowledge Chunks", doc_count)
        except:
            st.metric("Knowledge Chunks", "Ready")
    with col3:
        st.metric("Status", "✅ Online")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Display chat messages
    st.subheader("💬 Medical Consultation")
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Chat input
    prompt = st.chat_input("Ask about medical conditions, treatments, medications...")
    
    if prompt:
        # Add user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching medical knowledge..."):
                try:
                    MEDICAL_PROMPT_TEMPLATE = """You are medi_bot, an expert AI medical assistant. Use the medical context to answer the question.

Medical Context:
{context}

Question: {question}

Provide accurate medical information based only on the context:"""

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatGroq(
                            model_name="llama-3.1-8b-instant",
                            temperature=0.1,
                            groq_api_key=st.secrets["GROQ_API_KEY"],
                        ),
                        chain_type="stuff",
                        retriever=st.session_state.vectorstore.as_retriever(
                            search_kwargs={'k': 3}
                        ),
                        return_source_documents=True,
                        chain_type_kwargs={'prompt': set_custom_prompt(MEDICAL_PROMPT_TEMPLATE)}
                    )

                    response = qa_chain.invoke({'query': prompt})
                    result = response["result"]
                    source_documents = response["source_documents"]
                    
                    # Display response
                    st.markdown(result)
                    
                    # Display sources
                    with st.expander("📚 Source Documents"):
                        if source_documents:
                            for i, doc in enumerate(source_documents, 1):
                                source_file = doc.metadata.get('source', 'Unknown')
                                page_num = doc.metadata.get('page', 'N/A')
                                st.write(f"**Source {i}:** {source_file} (Page {page_num})")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.divider()
                    
                    # Add to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result})
                    
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Medical Information:**
- Always consult healthcare professionals
- This is for informational purposes only
- Verify critical medical information
""")