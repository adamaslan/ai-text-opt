#!/usr/bin/env python3
"""
FastAPI backend for ChromaDB RAG Chatbot
Exposes POST /chat and GET /collections
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from chroma_rag_chatbot import (
    ChromaRAGChatbot,
    ChromaRetriever,
    GeminiBackend,
    OllamaBackend,
    RetrievedDoc,
)

app = FastAPI(title="ChromaDB RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")  # "gemini" or "ollama"
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

_chatbot: Optional[ChromaRAGChatbot] = None


def get_chatbot() -> ChromaRAGChatbot:
    global _chatbot
    if _chatbot is None:
        retriever = ChromaRetriever()
        llm = OllamaBackend() if LLM_BACKEND == "ollama" else GeminiBackend()
        _chatbot = ChromaRAGChatbot(retriever=retriever, llm=llm, top_k=TOP_K)
    return _chatbot


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class SourceDoc(BaseModel):
    collection: str
    doc_id: str
    distance: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]


@app.get("/collections")
def list_collections() -> List[str]:
    try:
        return get_chatbot()._retriever.list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        bot = get_chatbot()
        top_k = req.top_k or TOP_K
        answer, docs = bot.ask(req.question, top_k=top_k)
        sources = [
            SourceDoc(
                collection=d.source_collection,
                doc_id=d.doc_id,
                distance=round(d.distance, 4),
                excerpt=d.text[:200],
            )
            for d in docs
        ]
        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
