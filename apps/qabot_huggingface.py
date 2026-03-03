# /// script
# dependencies = [
#   "python-dotenv",
#   "gradio>=5.0.0",
#   "langchain>=0.3.0",
#   "langchain-community",
#   "langchain-huggingface>=0.1.2",
#   "langchain-text-splitters",
#   "huggingface-hub>=0.28.1",
#   "faiss-cpu",
#   "pypdf",
#   "pydantic<2.10.0",
#   "accelerate>=0.34.0",
#   "transformers>=4.45.0",
#   "FlagEmbedding",
#   "torch"
# ]
# ///

import os, shutil, hashlib
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr

os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from FlagEmbedding import FlagReranker
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
    ChatHuggingFace
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# -------------------------------------------------------------------
# Environment + global clients
# -------------------------------------------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found. Check your .env file.")

# MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
#EMBED_MODEL_ID = "BAAI/bge-m3" # embedding dimension 1024, max tokens 8192
#EMBED_MODEL_ID = "sentence-transformers/all-mpnet-base-v2" # embedding dimension 768, max tokens 512
EMBED_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1" # embedding dimension 1024, max tokens 2048
RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
FAISS_PATH = "faiss_index"

# -------------------------------------------------------------------
# LLM, Embedding, and Reranker models initilization
# -------------------------------------------------------------------

# Use the model hosted on Hugging Face without downloading it locally
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(    
        repo_id=MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=1024,
    )
)

embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBED_MODEL_ID,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)

# Native BGE Reranker (Native library for score visibility)
native_reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

# -------------------------------------------------------------------
# Prompts & Formatting
# -------------------------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """You are a professional research assistant. Answer the question based ONLY on the provided context.

### CONTEXT:
{context}

### INSTRUCTIONS:
1. Use ONLY the information provided in the context above.
2. If the context does not contain the answer, state: "I'm sorry, but the provided document does not contain information to answer this question."
3. Use bullet points for structured data and keep a professional tone.
4. Reference page numbers when available.

Question: {question}

Helpful Answer:"""
)

def process_and_format_docs(docs):
    """Consolidated function to log scores and prepare LLM context."""
    if not docs:
        return "No relevant information found in the document."

    print(f"\n--- Reranker Report: {len(docs)} Docs Delivered to LLM ---")
    formatted_chunks = []
    
    for i, doc in enumerate(docs):
        score = doc.metadata.get("relevance_score", 0.0)
        snippet = doc.page_content[:70].replace('\n', ' ')
        print(f"Rank {i+1} [Score: {score:.4f}]: {snippet}...")
        
        # Prepare content with source info for the LLM
        page = doc.metadata.get('page_label', doc.metadata.get('page', 'N/A'))
        formatted_chunks.append(f"--- Document Source (Page {page}) ---\n{doc.page_content}")
        
    return "\n\n".join(formatted_chunks)

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def compute_pdf_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_saved_hash():
    hash_path = os.path.join(FAISS_PATH, "pdf.hash")
    if not os.path.exists(hash_path):
        return None
    with open(hash_path, "r") as f:
        return f.read().strip()

# -------------------------------------------------------------------
# Core RAG Logic
# -------------------------------------------------------------------

def retriever_qa(file, query, progress=gr.Progress()):
    if not file or not query:
        return "Ensure PDF is uploaded and question is typed."

    try:
        current_hash = compute_pdf_hash(file)
        saved_hash = load_saved_hash()

        if saved_hash == current_hash:
            progress(0.2, desc="Loading existing vector index...")
            vectorstore = FAISS.load_local(
                FAISS_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

        else:
            progress(0.1, desc="Analyzing new PDF...")
            loader = PyPDFLoader(file)
            docs = loader.load()
            progress(0.3, desc="Splitting text into larger chunks...")
            # Chunk size and chunk overlap values chosen to best fit research/technical pdfs
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            # Vector store
            progress(0.5, desc="Generating embeddings...")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            # Persist FAISS
            vectorstore.save_local(FAISS_PATH)
            with open(os.path.join(FAISS_PATH, "pdf.hash"), "w") as f:
                f.write(current_hash)

        # 1. Broad Retrieval (Vector Search)
        progress(0.7, desc="Searching vector database...")
        base_docs=vectorstore.as_retriever(search_kwargs={"k": 25}).invoke(query)

        # 2. Native Reranking (Cross-Encoder)
        progress(0.8, desc="Reranking for precision...")
        pairs = [[query, doc.page_content] for doc in base_docs]
        scores = native_reranker.compute_score(pairs)

        # 3. Score Attachment & Sorting
        for i, doc in enumerate(base_docs):
            doc.metadata["relevance_score"] = float(scores[i])

        # 4. Filter & Select Top 5 (Relevance Threshold of -10)
        reranked_docs = sorted(base_docs, key=lambda x: x.metadata["relevance_score"], reverse=True)[:5]
        reranked_docs = [d for d in reranked_docs if d.metadata["relevance_score"] > -10.0]
        
        # 5. LCEL Generation Pipeline
        progress(0.9, desc="LLM is synthesizing answer...")
        rag = (
            {
                "context": RunnableLambda(lambda _: process_and_format_docs(reranked_docs)),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag.invoke(query)

    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------------------------------------------
# UI
# -------------------------------------------------------------------

def reset_index():
    if Path(FAISS_PATH).exists():
        shutil.rmtree(FAISS_PATH)

    return None, "", ""

with gr.Blocks() as demo:
    gr.Markdown("# Professional RAG Console")
    gr.Markdown(f"### LCEL Pipe - LLM: **{MODEL_ID}**, Embeddings: **{EMBED_MODEL_ID}**, Reranker: **{RERANKER_MODEL_ID}**.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath")
            query_input = gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
            with gr.Row():
                reset_btn = gr.Button("Reset", variant="secondary")
                btn = gr.Button("Submit", variant="primary")
        answer_output = gr.Textbox(label="AI Analysis", lines=15)

    reset_btn.click(reset_index, inputs=None, outputs=[file_input, query_input, answer_output])
    btn.click(retriever_qa, [file_input, query_input], answer_output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())