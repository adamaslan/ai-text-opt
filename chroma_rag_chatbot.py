#!/usr/bin/env python3
"""
ChromaDB RAG Chatbot

Retrieval-Augmented Generation backed by your local ChromaDB collections.
Supports Google Gemini (default) or local Ollama as the LLM backend.

Designed for easy porting to Weaviate: swap out ChromaRetriever for a
WeaviateRetriever that implements the same VectorRetriever protocol and
nothing else changes.

Usage:
    python chroma_rag_chatbot.py                        # all collections, Gemini LLM
    python chroma_rag_chatbot.py --collection ideas     # single collection
    python chroma_rag_chatbot.py --llm ollama           # use local Ollama instead
    python chroma_rag_chatbot.py --top-k 8              # retrieve more context docs
    python chroma_rag_chatbot.py --list-collections     # show available collections

Requirements:
    pip install chromadb python-dotenv requests
    pip install langchain-google-genai   # for Gemini backend
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent
load_dotenv(dotenv_path=PROJECT_DIR / ".env")

CHROMA_DB_PATH = str(PROJECT_DIR / "chromadb_storage")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "dolphin-phi:2.7b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

PROMPT_TEMPLATE = """\
You are a thoughtful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say so honestly.
Cite specific details from the context when helpful.

Context:
{context}

Question: {question}

Answer:"""

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetrievedDoc:
    """A single retrieved document with its metadata."""
    text: str
    source_collection: str
    doc_id: str
    distance: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# VectorRetriever protocol  (swap ChromaRetriever → WeaviateRetriever here)
# ---------------------------------------------------------------------------

class VectorRetriever(ABC):
    """Abstract retriever — implement this for any vector store backend."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        """Return top_k documents most relevant to query."""
        ...

    @abstractmethod
    def list_collections(self) -> List[str]:
        """Return available collection names."""
        ...


# ---------------------------------------------------------------------------
# ChromaDB retriever
# ---------------------------------------------------------------------------

class ChromaRetriever(VectorRetriever):
    """Retrieves documents from one or more ChromaDB collections."""

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collections: Optional[List[str]] = None,
    ):
        import chromadb as _chromadb

        self._client = _chromadb.PersistentClient(path=db_path)
        all_names = [c.name for c in self._client.list_collections()]

        if not all_names:
            raise RuntimeError(f"No collections found in ChromaDB at {db_path}")

        if collections:
            missing = [c for c in collections if c not in all_names]
            if missing:
                raise ValueError(
                    f"Collections not found: {missing}. Available: {all_names}"
                )
            self._target = collections
        else:
            self._target = all_names

        print(f"  ChromaDB connected — searching: {self._target}")

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.list_collections()]

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        docs: List[RetrievedDoc] = []

        for name in self._target:
            collection = self._client.get_collection(name)
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(top_k, collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                logger.warning("Collection %s query failed: %s", name, e)
                continue

            ids = results["ids"][0]
            texts = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]

            for doc_id, text, meta, dist in zip(ids, texts, metas, dists):
                docs.append(
                    RetrievedDoc(
                        text=text,
                        source_collection=name,
                        doc_id=doc_id,
                        distance=dist,
                        metadata=meta or {},
                    )
                )

        # Sort by distance ascending (most relevant first), keep top_k overall
        docs.sort(key=lambda d: d.distance)
        return docs[:top_k]


# ---------------------------------------------------------------------------
# Weaviate retriever stub — fill this in when you're ready to port
# ---------------------------------------------------------------------------

class WeaviateRetriever(VectorRetriever):
    """
    Drop-in replacement for ChromaRetriever using Weaviate.

    To activate:
        1. pip install -U weaviate-client
        2. docker compose up -d
        3. Pass --retriever weaviate to the CLI (or instantiate directly)

    The rest of the chatbot (LLMs, prompts, CLI) is unchanged.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        collections: Optional[List[str]] = None,
    ):
        try:
            import weaviate as _weaviate
        except ImportError:
            raise ImportError("Run: pip install -U 'weaviate-client'")

        self._client = _weaviate.connect_to_local()
        self._collections = collections or [
            c.name for c in self._client.collections.list_all().values()
        ]
        print(f"  Weaviate connected — searching: {self._collections}")

    def list_collections(self) -> List[str]:
        return [c.name for c in self._client.collections.list_all().values()]

    def retrieve(self, query: str, top_k: int) -> List[RetrievedDoc]:
        docs: List[RetrievedDoc] = []

        for name in self._collections:
            collection = self._client.collections.get(name)
            try:
                results = collection.query.near_text(
                    query=query,
                    limit=top_k,
                    return_metadata=["distance"],
                )
            except Exception as e:
                logger.warning("Weaviate collection %s query failed: %s", name, e)
                continue

            for obj in results.objects:
                props = obj.properties
                docs.append(
                    RetrievedDoc(
                        text=props.get("text_content") or props.get("text_preview", ""),
                        source_collection=name,
                        doc_id=str(obj.uuid),
                        distance=obj.metadata.distance or 0.0,
                        metadata=props,
                    )
                )

        docs.sort(key=lambda d: d.distance)
        return docs[:top_k]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Abstract LLM — swap Gemini ↔ Ollama without touching the chat loop."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class GeminiBackend(LLMBackend):
    """Google Gemini via langchain-google-genai."""

    def __init__(
        self,
        model: str = GEMINI_MODEL,
        api_key: str = GOOGLE_API_KEY,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ):
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Add it to your .env file or use --llm ollama."
            )
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Run: pip install langchain-google-genai")

        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        self._model = model

    @property
    def name(self) -> str:
        return f"Gemini ({self._model})"

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt).content


class OllamaBackend(LLMBackend):
    """Local Ollama via direct HTTP — no extra dependencies."""

    def __init__(
        self,
        model: str = OLLAMA_LLM_MODEL,
        base_url: str = OLLAMA_BASE,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._verify()

    def _verify(self) -> None:
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Ollama not reachable at {self._base_url}: {e}\n"
                "Start it with: ollama serve"
            ) from e

    @property
    def name(self) -> str:
        return f"Ollama ({self._model})"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as e:
            return f"[Ollama error: {e}]"


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------

class ChromaRAGChatbot:
    """
    Wires a VectorRetriever to an LLMBackend.
    Swap either component without touching the chat loop.
    """

    def __init__(self, retriever: VectorRetriever, llm: LLMBackend, top_k: int = DEFAULT_TOP_K):
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k

    def ask(self, question: str) -> tuple[str, List[RetrievedDoc]]:
        """Returns (answer, retrieved_docs)."""
        docs = self._retriever.retrieve(question, self._top_k)

        if not docs:
            return "I couldn't find any relevant information in the knowledge base.", []

        context = "\n\n---\n\n".join(
            f"[{d.source_collection}] {d.text}" for d in docs
        )
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        answer = self._llm.generate(prompt)
        return answer, docs


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _print_sources(docs: List[RetrievedDoc], max_shown: int = 3) -> None:
    print("\n  Sources:")
    for i, doc in enumerate(docs[:max_shown], 1):
        excerpt = doc.text[:120].replace("\n", " ")
        print(f"    {i}. [{doc.source_collection}] dist={doc.distance:.4f}  {excerpt}...")


def _build_retriever(args: argparse.Namespace) -> VectorRetriever:
    collections = args.collection or None
    if args.retriever == "weaviate":
        return WeaviateRetriever(collections=collections)
    return ChromaRetriever(collections=collections)


def _build_llm(args: argparse.Namespace) -> LLMBackend:
    if args.llm == "ollama":
        return OllamaBackend(
            model=args.ollama_model,
            temperature=args.temperature,
        )
    return GeminiBackend(temperature=args.temperature)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChromaDB RAG Chatbot — Gemini or Ollama backed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--collection", "-c",
        nargs="+",
        metavar="NAME",
        help="Collection(s) to search. Defaults to all.",
    )
    parser.add_argument(
        "--llm",
        choices=["gemini", "ollama"],
        default="gemini",
        help="LLM backend (default: gemini)",
    )
    parser.add_argument(
        "--retriever",
        choices=["chroma", "weaviate"],
        default="chroma",
        help="Vector store backend (default: chroma)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        dest="top_k",
        help=f"Documents to retrieve per query (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"LLM temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_LLM_MODEL,
        help=f"Ollama model name (default: {OLLAMA_LLM_MODEL})",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="Print available collections and exit",
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Hide source documents after each answer",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print("\n" + "=" * 65)
    print("  ChromaDB RAG Chatbot")
    print("=" * 65)

    # Build retriever
    try:
        retriever = _build_retriever(args)
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"\nRetriever error: {e}")
        sys.exit(1)

    # List-only mode
    if args.list_collections:
        print("\nAvailable collections:")
        for name in retriever.list_collections():
            print(f"  - {name}")
        sys.exit(0)

    # Build LLM
    try:
        llm = _build_llm(args)
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"\nLLM error: {e}")
        sys.exit(1)

    chatbot = ChromaRAGChatbot(retriever=retriever, llm=llm, top_k=args.top_k)

    print(f"  LLM        : {llm.name}")
    print(f"  Retriever  : {args.retriever}")
    print(f"  Top-K      : {args.top_k}")
    print(f"\nType your question or 'exit' to quit.")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q", "bye"):
            print("Goodbye!")
            break

        try:
            answer, docs = chatbot.ask(query)
            print(f"\nAssistant: {answer}\n")
            if docs and not args.no_sources:
                _print_sources(docs)
            print()
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
