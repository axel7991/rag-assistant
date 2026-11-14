from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from chunking import chunk_text
from pdf_processing import extract_pdf_chunks
from uuid import uuid4
from pathlib import Path
import hashlib
from database import init_db, get_upload_by_key, reserve_upload, complete_upload, fail_upload
import json
import logging

logger = logging.getLogger(__name__)



load_dotenv()

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_data")
collection = chroma_client.get_or_create_collection(name="documents")


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
async def upload_file(
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(default=None)
):
    filename = file.filename or ""

    # Step 1: Validate PDF extension
    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    # Step 2: Validate file size and calculate SHA-256
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MiB

    file_size = 0
    sha256 = hashlib.sha256()

    while chunk := await file.read(1024 * 1024):
        file_size += len(chunk)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File exceeds the 10 MB limit"
            )

        sha256.update(chunk)

    file_hash = sha256.hexdigest()

    # Reset file pointer before saving
    await file.seek(0)

    # Step 3: Reserve idempotency key in SQLite
    if idempotency_key:
        init_db()

        reserved = reserve_upload(
            idempotency_key=idempotency_key,
            filename=filename,
            file_hash=file_hash
        )

        if not reserved:
            existing_upload = get_upload_by_key(
                idempotency_key
            )

            if existing_upload is None:
                raise HTTPException(
                    status_code=503,
                    detail="Unable to retrieve existing upload. Please retry."
                )

            # Prevent reuse with different PDF content
            if existing_upload["file_hash"] != file_hash:
                raise HTTPException(
                    status_code=409,
                    detail="Idempotency key already used for a different file"
                )

            # Upload currently being processed
            if existing_upload["status"] == "processing":
                raise HTTPException(
                    status_code=409,
                    detail="Upload is already being processed"
                )

            # Replay completed upload response
            if existing_upload["status"] == "completed":
                if existing_upload["response_json"]:
                    return json.loads(
                        existing_upload["response_json"]
                    )

                raise HTTPException(
                    status_code=500,
                    detail="Completed upload has no saved response"
                )

            # Failed upload recovery not yet implemented
            if existing_upload["status"] == "failed":
                raise HTTPException(
                    status_code=409,
                    detail="Previous upload failed; retry recovery not yet implemented"
                )

            raise HTTPException(
                status_code=409,
                detail="Idempotency key already exists"
            )

    # Step 4: Save and process PDF
    file_path = None
    indexing_succeeded = False

    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        safe_filename = f"{uuid4().hex}.pdf"
        file_path = upload_dir / safe_filename

        # Step 5: Save uploaded PDF
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 6: Extract PDF chunks
        chunks = extract_pdf_chunks(str(file_path))

        # Step 7: Generate embeddings and index chunks
        if chunks:
            texts = [chunk["text"] for chunk in chunks]

            embeddings = embedding_model.encode(
                texts,
                batch_size=32
            ).tolist()

            ids = []
            metadatas = []

            for chunk in chunks:
                # Deterministic ID:
                # SHA256(file_hash:page_number:chunk_index)
                chunk_id = hashlib.sha256(
                    (
                        f"{file_hash}:"
                        f"{chunk['page_number']}:"
                        f"{chunk['chunk_index']}"
                    ).encode()
                ).hexdigest()

                ids.append(chunk_id)

                metadatas.append({
                    "filename": filename,
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"]
                })

            # Upsert prevents duplicate IDs on retries
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )

        indexing_succeeded = True

        # Step 8: Prepare successful response
        response = {
            "message": "File uploaded and indexed successfully",
            "filename": filename,
            "total_chunks": len(chunks)
        }

        # Step 9: Mark upload as completed
        if idempotency_key:
            completed = complete_upload(
                idempotency_key=idempotency_key,
                response=response
            )

            if not completed:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to mark upload as completed"
                )

        return response

    except Exception as exc:
        logger.exception(
            "PDF upload failed for filename=%s",
            filename
        )

        # Mark processing failures in SQLite
        if idempotency_key and not indexing_succeeded:
            try:
                updated = fail_upload(
                    idempotency_key=idempotency_key,
                    error_message=str(exc)
                )

                if not updated:
                    logger.warning(
                        "Could not mark upload as failed: %s",
                        idempotency_key
                    )

            except Exception:
                logger.exception(
                    "Failed to update SQLite upload status"
                )

        # Preserve existing HTTP exceptions
        if isinstance(exc, HTTPException):
            raise

        raise HTTPException(
            status_code=500,
            detail="PDF processing failed"
        ) from exc

    finally:
        # Remove temporary PDF
        if file_path is not None:
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                logger.exception(
                    "Failed to remove temporary PDF: %s",
                    file_path
                )





@app.post("/ask")
async def ask_question(question: Question):
    question_embedding = embedding_model.encode(question.question).tolist()
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    
    relevant_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
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
    
    sources = []
    for i, chunk in enumerate(relevant_chunks):
        sources.append({
            "filename": metadatas[i]["filename"],
            "chunk_index": metadatas[i]["chunk_index"],
            "preview": chunk[:150] + "..."
        })

    return {
        "question": question.question,
        "answer": message.content[0].text,
        "sources": sources
    }