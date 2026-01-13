#!/usr/bin/env python3
"""
Gemini RAG Chatbot - Retrieval-Augmented Generation with Google Gemini API

This chatbot combines:
- High-quality MiniLM embeddings (384D) for semantic search
- FAISS vector store for fast retrieval
- Google Gemini API for intelligent response generation
- Modern LangChain Runnable API for orchestration

Usage:
    python gemini_rag_chatbot.py

Requirements:
    - Set GOOGLE_API_KEY in .env file
    - Run: setup_embeddings.sh (to generate embeddings if missing)
    - Run: pip install python-dotenv langchain-google-genai langchain-community
"""

import os
import sys
import ast
from typing import List, Tuple
from pathlib import Path

# Environment & Configuration
from dotenv import load_dotenv

# Data Processing
import pandas as pd
import numpy as np

# LangChain Components (modern Runnable API)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


# ============================================================================
# Configuration
# ============================================================================

class ChatbotConfig:
    """Configuration loaded from .env file with defaults"""

    # API Configuration
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

    # Data Paths (relative to script location)
    PROJECT_DIR = Path(__file__).parent
    EMBEDDINGS_CSV: str = "embeddings/151qa2_with_embeddings.csv"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Retriever Settings
    TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))
    SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.3"))

    # Generation Settings
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))


# Prompt template for RAG
PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context.

Context from knowledge base:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- Be concise and direct in your response
- If the context doesn't contain relevant information, say "I don't have enough information in the knowledge base to answer that question."
- Cite specific details from the context when possible

Answer:"""


# ============================================================================
# Embedding Loading
# ============================================================================

def load_embeddings_from_csv(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """
    Load text and embeddings from CSV file.

    Args:
        csv_path: Path to CSV file with 'text' and 'Embeddings' columns

    Returns:
        Tuple of (texts_list, embeddings_array)

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        AssertionError: If embedding dimensions don't match expected 384D
    """
    # Construct absolute path
    abs_path = ChatbotConfig.PROJECT_DIR / csv_path

    if not abs_path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {abs_path}\n"
            f"Run: cd {ChatbotConfig.PROJECT_DIR} && bash setup_embeddings.sh"
        )

    # Read CSV
    df = pd.read_csv(abs_path)

    # Validate columns
    if "text" not in df.columns or "Embeddings" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'Embeddings' columns")

    # Extract texts
    texts = df["text"].tolist()

    # Parse embeddings from string representation
    embeddings = []
    for i, emb_str in enumerate(df["Embeddings"]):
        try:
            emb = ast.literal_eval(emb_str)  # Parse "[0.1, 0.2, ...]"
            embeddings.append(emb)
        except (ValueError, SyntaxError) as e:
            raise ValueError(
                f"Failed to parse embedding at row {i}: {str(e)}"
            )

    embeddings_np = np.array(embeddings, dtype=np.float32)

    # Validate dimensions
    expected_dim = 384
    if embeddings_np.shape[1] != expected_dim:
        raise AssertionError(
            f"Expected {expected_dim}D embeddings, got {embeddings_np.shape[1]}D"
        )

    return texts, embeddings_np


# ============================================================================
# Vector Store Creation
# ============================================================================

def create_vector_store(texts: List[str], embeddings: np.ndarray) -> FAISS:
    """
    Create FAISS vector store from pre-computed embeddings.

    Args:
        texts: List of text documents
        embeddings: Pre-computed embeddings array (already normalized)

    Returns:
        FAISS vector store configured for similarity search
    """
    # Initialize embedding model for query encoding
    embedding_model = HuggingFaceEmbeddings(
        model_name=ChatbotConfig.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create FAISS index from pre-computed embeddings
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=embedding_model,
        normalize_L2=False,  # Already unit-normalized
    )

    return vector_store


# ============================================================================
# LLM Setup
# ============================================================================

def create_gemini_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize Gemini LLM.

    Returns:
        ChatGoogleGenerativeAI instance configured for RAG

    Raises:
        ValueError: If GOOGLE_API_KEY is not set
    """
    if not ChatbotConfig.GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment.\n"
            "Please create a .env file with: GOOGLE_API_KEY=your_api_key_here\n"
            "See .env.example for template."
        )

    return ChatGoogleGenerativeAI(
        model=ChatbotConfig.GEMINI_MODEL,
        google_api_key=ChatbotConfig.GOOGLE_API_KEY,
        temperature=ChatbotConfig.TEMPERATURE,
        max_output_tokens=ChatbotConfig.MAX_TOKENS,
    )


def create_prompt() -> PromptTemplate:
    """Create the RAG prompt template."""
    return PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )


# ============================================================================
# RAG Chain Assembly (Modern Runnable API)
# ============================================================================

def create_rag_chain(vector_store: FAISS, llm: ChatGoogleGenerativeAI):
    """
    Assemble the RAG chain using modern LangChain Runnable API.

    Args:
        vector_store: FAISS vector store with embeddings
        llm: Gemini LLM instance

    Returns:
        Runnable chain ready for queries
    """
    # Create retriever
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": ChatbotConfig.TOP_K,
            "score_threshold": ChatbotConfig.SCORE_THRESHOLD,
        }
    )

    # Create prompt
    prompt = create_prompt()

    # Modern Runnable API composition
    def format_docs(docs):
        """Format retrieved documents for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Create chain: input -> retriever -> format -> prompt -> llm -> output
    chain = (
        RunnableParallel(
            context=retriever | (lambda docs: format_docs(docs)),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ============================================================================
# Chat Interface
# ============================================================================

def run_chat_interface(chain, retriever) -> None:
    """
    Run interactive CLI chat loop.

    Args:
        chain: Configured RAG Runnable chain
        retriever: Retriever for getting source documents
    """
    print("\n" + "=" * 70)
    print("🤖 Gemini RAG Chatbot - Interactive Mode")
    print("=" * 70)
    print(f"Model: {ChatbotConfig.GEMINI_MODEL}")
    print(f"Knowledge Base: {ChatbotConfig.EMBEDDINGS_CSV}")
    print(f"Retrieval: Top {ChatbotConfig.TOP_K} results")
    print("\nType your question or 'exit' to quit.")
    print("=" * 70 + "\n")

    while True:
        try:
            # Get user input
            query = input("🤔 You: ").strip()

            # Check for exit commands
            if query.lower() in ["exit", "quit", "q", "bye"]:
                print("\n👋 Goodbye!")
                break

            # Validate input
            if not query:
                print("⚠️  Please enter a question.\n")
                continue

            # Execute query
            print("\n🤖 Gemini: ", end="", flush=True)
            response = chain.invoke(query)
            print(response)

            # Get and display source documents
            try:
                docs = retriever.get_relevant_documents(query)
                if docs:
                    print("\n📚 Sources:")
                    for i, doc in enumerate(docs[:3], 1):
                        excerpt = doc.page_content[:100].replace("\n", " ")
                        print(f"  {i}. {excerpt}...")
            except Exception as e:
                # Sources are optional
                pass

            print()  # Blank line for readability

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'exit' to quit.\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> None:
    """
    Main entry point for the chatbot application.

    Initializes:
    1. Environment configuration
    2. Embeddings and vector store
    3. Gemini LLM
    4. RAG chain
    5. Chat interface
    """
    try:
        # Load environment variables from .env
        env_path = ChatbotConfig.PROJECT_DIR / ".env"
        load_dotenv(dotenv_path=env_path)

        # Validate API key
        if not ChatbotConfig.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found in .env file.\n"
                "Please set: GOOGLE_API_KEY=your_api_key_here\n"
                "See .env.example for template."
            )

        print("\n" + "=" * 70)
        print("Initializing Gemini RAG Chatbot...")
        print("=" * 70)

        # Step 1: Load embeddings
        print("\n📚 Loading embeddings...")
        texts, embeddings = load_embeddings_from_csv(ChatbotConfig.EMBEDDINGS_CSV)
        print(f"   ✓ Loaded {len(texts)} documents")
        print(f"   ✓ Embedding dimension: {embeddings.shape[1]}D")

        # Step 2: Create vector store
        print("\n🔍 Creating vector store...")
        vector_store = create_vector_store(texts, embeddings)
        print(f"   ✓ FAISS index created with {vector_store.index.ntotal} entries")

        # Step 3: Initialize Gemini LLM
        print("\n🧠 Connecting to Gemini API...")
        llm = create_gemini_llm()
        print(f"   ✓ Connected to {ChatbotConfig.GEMINI_MODEL}")

        # Step 4: Build RAG chain
        print("\n⛓️  Building RAG pipeline...")
        chain, retriever = create_rag_chain(vector_store, llm)
        print("   ✓ RAG chain ready")

        # Step 5: Start chat interface
        run_chat_interface(chain, retriever)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Configuration Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
