# backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import faiss
import numpy as np
import io

app = FastAPI()

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load LLM
llm_model_id = "HuggingFaceH4/zephyr-7b-alpha"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_id)
llm_pipeline = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer)

# Global state
pdf_chunks = []
faiss_index = None

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_chunks, faiss_index

    # Read PDF content
    content = await file.read()
    doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
    full_text = "\n".join([page.get_text() for page in doc])

    # Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    pdf_chunks = splitter.split_text(full_text)

    # Embed and store in FAISS
    embeddings = embed_model.encode(pdf_chunks)
    dim = embeddings[0].shape[0]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(np.array(embeddings))

    return {"message": "PDF uploaded successfully", "chunks": len(pdf_chunks)}

@app.post("/ask_question")
async def ask_question(payload: dict):
    global pdf_chunks, faiss_index

    question = payload.get("question", "")
    if not pdf_chunks or not faiss_index:
        return {"answer": "Please upload a PDF first."}

    # Embed question
    q_embedding = embed_model.encode([question])
    D, I = faiss_index.search(np.array(q_embedding), k=5)
    context = "\n\n".join([pdf_chunks[i] for i in I[0]])

    # Prompt to LLM
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    result = llm_pipeline(prompt, max_new_tokens=256, do_sample=False)
    answer = result[0]['generated_text'].split("Answer:")[-1].strip()

    return {"answer": answer}
