# DocQuery — AI-Powered RAG Document Assistant

Upload any PDF and ask questions in plain English. Get accurate, cited answers powered by Retrieval Augmented Generation (RAG).

🌐 **Live API:** [api.webwatchhq.com/docs](https://api.webwatchhq.com/docs)

## What it does

- Upload any PDF document (contracts, reports, research papers, manuals)
- Ask questions in plain English
- Get accurate answers with **source citations** showing exactly which part of the document was used
- Powered by RAG — answers only from your document, no hallucinations

## Tech Stack

- **Python + FastAPI** — REST API backend
- **PyPDF** — PDF text extraction
- **Sentence Transformers** — local embedding model (all-MiniLM-L6-v2)
- **ChromaDB** — persistent vector database for semantic search
- **Claude API (Anthropic)** — AI answer generation from retrieved context
- **Docker** — containerized deployment
- **AWS EC2** — 24/7 cloud deployment
- **Nginx + Let's Encrypt** — production web server with HTTPS
- **GitHub Actions** — CI/CD auto-deployment on push

## How RAG works