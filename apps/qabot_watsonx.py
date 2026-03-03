# /// script
# dependencies = [
#   "python-dotenv",
#   "gradio>=6.0.0",
#   "langchain>=1.0.0",
#   "langchain-ibm>=1.0.0",
#   "langchain-classic",
#   "langchain-text-splitters",
#   "langchain-community",
#   "langchain_chroma",
#   "chromadb",
#   "pypdf"
# ]
# ///

import os, hashlib, shutil, gc, stat, uuid
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings, WatsonxRerank
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------------------------
# Environment + Global clients
# -------------------------------------------------------------------

load_dotenv()
API_KEY = os.getenv("WATSONX_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
MODEL_ID = "ibm/granite-4-h-small"
EMBED_MODEL_ID = "ibm/slate-125m-english-rtrvr-v2"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-minilm-l-12-v2"
CHROMA_PATH = "chroma_db_watsonx"

# Global reference to manage Windows file locks
vectordb = None

# -------------------------------------------------------------------
# LLM, Embedding, and Reranker functions
# -------------------------------------------------------------------

def get_llm():
    return WatsonxLLM(model_id=MODEL_ID, url=URL, apikey=API_KEY, project_id=PROJECT_ID,
        params={GenParams.MAX_NEW_TOKENS: 1024, GenParams.TEMPERATURE: 0.5,
            GenParams.DECODING_METHOD: "greedy"})

def get_embeddings():
    return WatsonxEmbeddings(model_id=EMBED_MODEL_ID, url=URL,
        apikey=API_KEY, project_id=PROJECT_ID)

def get_reranker():
    return WatsonxRerank(model_id=RERANKER_MODEL_ID, url=URL,
            apikey=API_KEY, project_id=PROJECT_ID, params={"top_n": 15})

# -------------------------------------------------------------------
# Prompt
# -------------------------------------------------------------------

prompt_template = """<|system|>
You are an intelligent assistant helping to answer questions using retrieved context. 
Answer the question based ONLY on the context below. 
If the answer is not in the context, say that you do not know.

### STRICT RULES:
1. No repetition: do not repeat facts across different sections.
2. No meta-talk: do not include "Thinking" or "Enough thinking".

<documents>
{context}
</documents>

<|user|>
Provide a structured, high-density analysis and summary of: {question}

<|assistant|>
""".strip()

prompt = ChatPromptTemplate.from_template(prompt_template)

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def compute_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def load_saved_hash():
    hash_path = os.path.join(CHROMA_PATH, "pdf.hash")
    if not os.path.exists(hash_path):
        return None
    with open(hash_path, "r") as f:
        return f.read().strip()

def remove_readonly(func, path, excinfo):
    """Force-clears read-only flags to allow deletion on Windows."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def safe_clear_chroma():
    """Forces a clean slate by purging the directory and clearing memory."""
    global vectordb
    vectordb = None
    gc.collect()
    
    if Path(CHROMA_PATH).exists():
        try:
            shutil.rmtree(CHROMA_PATH, onerror=remove_readonly)
        except:
            # Fallback for Windows locks
            try:
                temp_name = f"trash_{uuid.uuid4().hex[:6]}"
                os.rename(CHROMA_PATH, temp_name)
                shutil.rmtree(temp_name, ignore_errors=True)
            except:
                pass

# -------------------------------------------------------------------
# Core RAG Logic
# -------------------------------------------------------------------

def retriever_qa(file, query, progress=gr.Progress()):
    global vectordb
    if not file or not query:
        return "Please upload a PDF and ask a question."

    try:
        current_hash = compute_hash(file)
        # Use the hash as a unique collection name to prevent cross-talk
        collection_id = f"pdf_{current_hash[:12]}"
        hash_file = Path(CHROMA_PATH) / "pdf.hash"
        needs_rebuild = True

        if hash_file.exists():
            if hash_file.read_text().strip() == current_hash:
                needs_rebuild = False

        if needs_rebuild:
            progress(0.1, desc="New PDF detected. Clearing old index...")
            safe_clear_chroma()

            progress(0.3, desc="Parsing new PDF...")
            loader = PyPDFLoader(file)

            # Chunk size and chunk overlap values chosen to best fit research/technical pdfs
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
            chunks = splitter.split_documents(loader.load())
            
            progress(0.5, desc="Building new vector index...")
            vectordb = Chroma.from_documents(
                documents=chunks, 
                embedding=get_embeddings(), 
                persist_directory=CHROMA_PATH,
                collection_name=collection_id
            )

            # Ensure the directory exists and save the hash
            os.makedirs(CHROMA_PATH, exist_ok=True)
            Path(hash_file).write_text(current_hash)    
        
        elif vectordb is None:
            progress(0.2, desc="Loading existing Watsonx index...")
            # If hash matches and folder exists, just load
            vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embeddings(), 
                collection_name=collection_id)

        progress(0.7, desc="Watsonx Reranker is prioritizing context...")
        base_retriever = vectordb.as_retriever(search_kwargs={"k": 30})
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=get_reranker(), 
            base_retriever=base_retriever
        )

        progress(0.9, desc="Granite is generating response...")
        docs = compression_retriever.invoke(query)
        
        context = "\n\n".join([f"<doc page='{d.metadata.get('page','?')}'>{d.page_content}</doc>" for d in docs])

        formatted_prompt = prompt.format(
            context=context,
            question=query
        )
        progress(0.9, desc="Generating Response...")
        return get_llm().invoke(formatted_prompt)

    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

def reset_index():
    safe_clear_chroma()
    return None, "", ""

with gr.Blocks() as demo:
    gr.Markdown("# Watsonx Professional RAG Console")
    gr.Markdown(f"### LLM: **{MODEL_ID}**, Embeddings: **{EMBED_MODEL_ID}**, Reranker: **{RERANKER_MODEL_ID}**.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF file", file_count="single", file_types=['.pdf'], type="filepath")
            query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
            with gr.Row():
                reset_btn = gr.Button("Reset", variant="secondary")
                btn = gr.Button("Analyze", variant="primary")
        answer_output = gr.Textbox(label="AI Analysis", lines=15)
    
    reset_btn.click(reset_index, inputs=None, outputs=[file_input, query_input, answer_output])
    btn.click(retriever_qa, [file_input, query_input], answer_output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())