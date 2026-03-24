#!/usr/bin/env python3
"""
Test script for Gemini RAG Chatbot
Tests all major components without requiring interactive input
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

import gemini_rag_chatbot as rag


def test_environment_loading():
    """Test 1: Environment variable loading from .env"""
    print("\n" + "=" * 70)
    print("TEST 1: Environment Variable Loading")
    print("=" * 70)

    env_path = rag.ChatbotConfig.PROJECT_DIR / ".env"
    print(f"Loading .env from: {env_path}")
    print(f"File exists: {env_path.exists()}")

    load_dotenv(dotenv_path=env_path)

    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"✓ GOOGLE_API_KEY loaded: {'***' + api_key[-4:] if api_key else 'NOT FOUND'}")

    gemini_model = os.getenv("GEMINI_MODEL")
    print(f"✓ GEMINI_MODEL: {gemini_model}")

    top_k = os.getenv("RAG_TOP_K")
    print(f"✓ RAG_TOP_K: {top_k}")

    if not api_key:
        print("❌ GOOGLE_API_KEY not found!")
        return False

    print("✅ Environment loading PASSED")
    return True


def test_embeddings_loading():
    """Test 2: Load embeddings from CSV"""
    print("\n" + "=" * 70)
    print("TEST 2: Embeddings CSV Loading")
    print("=" * 70)

    csv_path = rag.ChatbotConfig.EMBEDDINGS_CSV
    abs_path = rag.ChatbotConfig.PROJECT_DIR / csv_path
    print(f"Loading from: {abs_path}")
    print(f"File exists: {abs_path.exists()}")

    try:
        texts, embeddings = rag.load_embeddings_from_csv(csv_path)
        print(f"✓ Loaded {len(texts)} documents")
        print(f"✓ Embeddings shape: {embeddings.shape}")
        print(f"✓ First text (truncated): {texts[0][:100]}...")
        print(f"✓ First embedding (first 5 dims): {embeddings[0][:5]}")
        print("✅ Embeddings loading PASSED")
        return True, texts, embeddings
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return False, None, None


def test_vector_store_creation(texts, embeddings):
    """Test 3: Create FAISS vector store"""
    print("\n" + "=" * 70)
    print("TEST 3: Vector Store Creation")
    print("=" * 70)

    try:
        vector_store = rag.create_vector_store(texts, embeddings)
        print(f"✓ Vector store created")
        print(f"✓ FAISS index entries: {vector_store.index.ntotal}")
        print(f"✓ Index dimension: {vector_store.index.d}")
        print("✅ Vector store creation PASSED")
        return True, vector_store
    except Exception as e:
        print(f"❌ Failed to create vector store: {e}")
        return False, None


def test_llm_initialization():
    """Test 4: Initialize Gemini LLM"""
    print("\n" + "=" * 70)
    print("TEST 4: LLM Initialization")
    print("=" * 70)

    try:
        llm = rag.create_gemini_llm()
        print(f"✓ LLM initialized: {rag.ChatbotConfig.GEMINI_MODEL}")
        print(f"✓ Temperature: {rag.ChatbotConfig.TEMPERATURE}")
        print(f"✓ Max tokens: {rag.ChatbotConfig.MAX_TOKENS}")
        print("✅ LLM initialization PASSED")
        return True, llm
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return False, None


def test_rag_chain_assembly(vector_store, llm):
    """Test 5: Assemble RAG chain"""
    print("\n" + "=" * 70)
    print("TEST 5: RAG Chain Assembly")
    print("=" * 70)

    try:
        chain, retriever = rag.create_rag_chain(vector_store, llm)
        print(f"✓ RAG chain created")
        print(f"✓ Retriever configured with top_k={rag.ChatbotConfig.TOP_K}")
        print(f"✓ Score threshold: {rag.ChatbotConfig.SCORE_THRESHOLD}")
        print("✅ RAG chain assembly PASSED")
        return True, chain, retriever
    except Exception as e:
        print(f"❌ Failed to assemble RAG chain: {e}")
        return False, None, None


def test_retrieval(retriever):
    """Test 6: Test retriever functionality"""
    print("\n" + "=" * 70)
    print("TEST 6: Retrieval Functionality")
    print("=" * 70)

    try:
        # Test with a simple query
        test_query = "question"
        docs = retriever.get_relevant_documents(test_query)
        print(f"✓ Retrieved {len(docs)} documents for query: '{test_query}'")
        if docs:
            print(f"✓ Top result (truncated): {docs[0].page_content[:100]}...")
        print("✅ Retrieval PASSED")
        return True
    except Exception as e:
        print(f"❌ Retrieval failed: {e}")
        return False


def test_full_pipeline():
    """Run all tests"""
    print("\n" + "🔧 " * 35)
    print("GEMINI RAG CHATBOT - FULL TEST SUITE")
    print("🔧 " * 35)

    results = {}

    # Test 1: Environment
    results['environment'] = test_environment_loading()
    if not results['environment']:
        print("\n❌ Cannot proceed without environment variables!")
        return False

    # Test 2: Embeddings
    success, texts, embeddings = test_embeddings_loading()
    results['embeddings'] = success
    if not success:
        print("\n❌ Cannot proceed without embeddings!")
        return False

    # Test 3: Vector Store
    success, vector_store = test_vector_store_creation(texts, embeddings)
    results['vector_store'] = success
    if not success:
        print("\n❌ Cannot proceed without vector store!")
        return False

    # Test 4: LLM
    success, llm = test_llm_initialization()
    results['llm'] = success
    if not success:
        print("\n❌ Cannot proceed without LLM!")
        return False

    # Test 5: RAG Chain
    success, chain, retriever = test_rag_chain_assembly(vector_store, llm)
    results['rag_chain'] = success
    if not success:
        print("\n❌ Cannot proceed without RAG chain!")
        return False

    # Test 6: Retrieval
    results['retrieval'] = test_retrieval(retriever)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper():20s} {status}")

    all_passed = all(results.values())
    print("=" * 70)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Chatbot is ready to use.")
        print("\nTo run the interactive chatbot, use:")
        print("  python gemini_rag_chatbot.py")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
