from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="documents")

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

app = FastAPI(title="RAG Document Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG Assistant API is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    extracted_text = extract_text_from_pdf(file_path)
    chunks = chunk_text(extracted_text)
    
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            ids=[f"{file.filename}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"filename": file.filename, "chunk_index": i}]
        )
    
    return {
        "message": "File uploaded and indexed successfully",
        "filename": file.filename,
        "total_chunks": len(chunks)
    }

@app.post("/ask")
async def ask_question(question: Question):
    question_embedding = embedding_model.encode(question.question).tolist()
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    
    relevant_chunks = results["documents"][0]
    context = "\n\n".join(relevant_chunks)
    
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Answer the question based ONLY on the context below. If the answer isn't in the context, say so.

CONTEXT:
{context}

QUESTION:
{question.question}

ANSWER:"""
        }]
    )
    
    return {
        "question": question.question,
        "answer": message.content[0].text,
        "sources_used": len(relevant_chunks)
    }