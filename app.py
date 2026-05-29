import streamlit as st
import tempfile
import os
from pathlib import Path

# Import your top-level façade
from rag_connector import RAGConnector

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Multimodal RAG", page_icon="🧠", layout="wide")
st.title("🧠 Multimodal RAG Assistant")
st.markdown("Upload documents, images, audio, or video, and ask questions about them!")

# --- 2. SESSION STATE SETUP ---
# Initialize the RAG connector once so it doesn't reload on every UI interaction
if "rag" not in st.session_state:
    with st.spinner("Initializing Local Models and Vector DB..."):
        st.session_state.rag = RAGConnector()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. SIDEBAR: DATA INGESTION ---
with st.sidebar:
    st.header("📥 Ingest Data")
    st.info("Supported: PDFs, DOCX, Images, Video, Audio, and Code.")
    
    uploaded_file = st.file_uploader("Upload a file to the knowledge base:")
    
    if uploaded_file is not None:
        if st.button("Process & Index File"):
            with st.spinner(f"Ingesting `{uploaded_file.name}` (This might take a while for video/audio)..."):
                # Save the uploaded file temporarily so the pipeline can read it from disk
                file_extension = Path(uploaded_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    # Run your ingestion and indexing pipeline
                    st.session_state.rag.index(tmp_path)
                    st.success(f"✅ Successfully indexed: {uploaded_file.name}!")
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                finally:
                    # Clean up the temp file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 4. MAIN CHAT INTERFACE ---
# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new questions
if prompt := st.chat_input("Ask a question based on your uploaded data..."):
    
    # Show user question instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the RAG response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and reasoning..."):
            try:
                # Query the backend.
                # Pass everything BEFORE the current question as history so the
                # backend can rewrite follow-ups into standalone search queries.
                results = st.session_state.rag.query(
                    question=prompt,
                    top_k=5,
                    chat_history=st.session_state.messages[:-1],
                )
                answer = results["answer"]
                sources = results["sources"]

                # Render the answer
                st.markdown(answer)

                # Render the sources in an expander
                if sources:
                    with st.expander("📚 View Sources used for this answer"):
                        # If the question was rewritten for retrieval, show it.
                        search_query = results.get("search_query", prompt)
                        if search_query != prompt:
                            st.caption(f"🔄 **Rewritten query:** _{search_query}_")
                        for i, src in enumerate(sources, 1):
                            meta = src.get("metadata", {})
                            source_name = meta.get('source', 'Unknown File')
                            modality = meta.get('modality', 'Unknown')
                            st.caption(
                                f"**[{i}] {source_name}** ({modality}) "
                                f"— RRF Score: `{src['rrf_score']:.3f}`"
                            )
                            # You can optionally print a snippet of the chunk text here too!
                
                # Save to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error during retrieval: {e}")
