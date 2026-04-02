#!/usr/bin/env python3
"""
Ingest Trader-QA docs into Zilliz Cloud via LlamaIndex
Run with: mamba activate fin-ai1 && python ingest.py
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from pymilvus import MilvusClient, DataType

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

DOCS_ROOT = Path(__file__).parent.parent / "docs" / "trader-qa"

FILES = [
    "trader-profiles-updated.md",
    "doc-1-options-and-swing-trading.md",
    "doc-2-industries-and-sectors.md",
    "doc-3-recent-news-part1.md",
    "doc-4-recent-news-part2.md",
    "REMAINING-QA-OUTLINE.md",
    "vde-xle-war-end-qa.md",
]

# API Keys
ZILLIZ_CLOUD_URI = os.getenv("ZILLIZ_CLOUD_URI")
ZILLIZ_API_KEY = os.getenv("ZILLIZ_API_KEY")
ZILLIZ_COLLECTION = os.getenv("ZILLIZ_COLLECTION", "Nu-Fin1")

# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

required_keys = {
    "ZILLIZ_CLOUD_URI": ZILLIZ_CLOUD_URI,
    "ZILLIZ_API_KEY": ZILLIZ_API_KEY,
}

missing = [k for k, v in required_keys.items() if not v]
if missing:
    print(f"❌ Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

if not DOCS_ROOT.exists():
    print(f"❌ Docs directory not found: {DOCS_ROOT}")
    sys.exit(1)

print(f"✅ All environment variables set")
print(f"📂 Docs directory: {DOCS_ROOT}")
print(f"📦 Collection: {ZILLIZ_COLLECTION}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load documents from markdown files
# ─────────────────────────────────────────────────────────────────────────────

async def load_documents():
    """Load and parse markdown files using LlamaParse."""
    print("Step 1: Parsing documents with LlamaParse...")
    print()

    parser = LlamaParse(api_key=os.getenv("LLAMA_CLOUD_API_KEY"), result_type="markdown")
    documents = []

    for file_name in FILES:
        file_path = DOCS_ROOT / file_name
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}, skipping...")
            continue

        print(f"📄 Parsing {file_name}...")
        try:
            # LlamaParse intelligently parses the markdown, preserving structure
            result = await parser.aload_data(str(file_path))

            # result is typically a list of Documents or a single Document
            if isinstance(result, list):
                for doc in result:
                    doc.metadata["source_file"] = file_name
                documents.extend(result)
                print(f"   ✅ Parsed into {len(result)} documents")
            else:
                result.metadata["source_file"] = file_name
                documents.append(result)
                print(f"   ✅ Parsed successfully")
        except Exception as e:
            print(f"   ❌ Error parsing {file_name}: {e}")
            # Fallback: load file as plain text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(text=content, metadata={"source_file": file_name})
                documents.append(doc)
                print(f"   ⚠️  Fell back to plain text loading ({len(content)} chars)")
            except Exception as fallback_e:
                print(f"   ❌ Fallback also failed: {fallback_e}")
                continue

    print()
    print(f"✅ Parsed {len(documents)} total documents")
    print()
    return documents

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Create ingestion pipeline and process documents
# ─────────────────────────────────────────────────────────────────────────────

def create_and_run_pipeline(documents):
    """Create LlamaIndex ingestion pipeline and process documents."""
    print("Step 2: Processing documents through pipeline...")
    print()

    # Pipeline configuration: chunking only
    splitter = SentenceSplitter(chunk_size=400, chunk_overlap=50)
    pipeline = IngestionPipeline(transformations=[splitter])

    print("   ✅ Pipeline created with:")
    print("      • SentenceSplitter (400 tokens, 50 overlap)")
    print()

    # Process documents
    print("   Processing chunks...")
    nodes = pipeline.run(documents=documents)

    print(f"   ✅ Pipeline produced {len(nodes)} nodes")
    print()
    return nodes

# ─────────────────────────────────────────────────────────────────────────────
# Step 2.5: Create collection with server-side embedding function
# ─────────────────────────────────────────────────────────────────────────────

def ensure_collection_with_embedding(client, drop_if_exists=True):
    """Create collection with built-in embedding function.

    Args:
        client: MilvusClient instance
        drop_if_exists: If True, drop existing empty collection and recreate with function.
                       This ensures the Function is properly attached.
    """
    print("Step 2.5: Setting up collection schema with server-side embedding...")
    print()

    # Check if collection exists
    existing_collections = client.list_collections()
    if ZILLIZ_COLLECTION in existing_collections:
        if drop_if_exists:
            print(f"   🗑️  Dropping existing collection '{ZILLIZ_COLLECTION}'...")
            try:
                client.drop_collection(ZILLIZ_COLLECTION)
                print(f"   ✅ Dropped successfully. Will recreate with embedding function.")
            except Exception as e:
                print(f"   ⚠️  Could not drop collection: {e}")
                print(f"   ℹ️  Using existing collection (may not have embedding function)")
                return True
        else:
            print(f"   ℹ️  Collection '{ZILLIZ_COLLECTION}' already exists")
            return True

    # Create schema with embedding function
    try:
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

        # Add fields
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="text_content", datatype=DataType.VARCHAR, max_length=8192)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=768)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=256)

        # Add the embedding function - Zilliz will auto-vectorize text_content
        # Using specific function name for easier debugging and query configuration
        schema.add_function(
            function_name="trader_qa_embedder",
            function_type="text_embedding",
            input_field_names=["text_content"],
            output_field_names=["embedding"],
            model_name="BAAI/bge-base-en-v1.5",
        )

        print("   ✅ Schema defined:")
        print("      • text_content (VARCHAR) → input to embedding function")
        print("      • embedding (768-dim FLOAT_VECTOR) → auto-generated by Zilliz")
        print("      • source_file (VARCHAR) → metadata")
        print()

        # Create index
        print("   📋 Creating AUTOINDEX on embedding field...")
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            metric_type="COSINE",
            index_type="AUTOINDEX"
        )

        # Create collection
        client.create_collection(
            collection_name=ZILLIZ_COLLECTION,
            schema=schema,
            index_params=index_params,
        )

        print(f"   ✅ Collection '{ZILLIZ_COLLECTION}' created with server-side embedding")
        print()
        return True

    except Exception as e:
        print(f"   ❌ Failed to create collection: {e}")
        import traceback
        traceback.print_exc()
        return False

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Ingest nodes into Zilliz Cloud
# ─────────────────────────────────────────────────────────────────────────────

def ingest_into_zilliz(nodes):
    """Insert nodes into Zilliz Cloud. Zilliz will auto-embed via the Function."""
    print("Step 3: Ingesting into Zilliz Cloud (server-side embedding)...")
    print()

    # Connect to Zilliz
    try:
        client = MilvusClient(uri=ZILLIZ_CLOUD_URI, token=ZILLIZ_API_KEY)
        print(f"   ✅ Connected to Zilliz Cloud")
    except Exception as e:
        print(f"   ❌ Failed to connect to Zilliz: {e}")
        return False

    # Ensure collection exists with embedding function
    if not ensure_collection_with_embedding(client):
        return False

    # Prepare data for insertion
    data_to_insert = []
    for node in nodes:
        row = {
            "text_content": node.get_content(),
            "source_file": node.metadata.get("source_file", "unknown"),
        }
        data_to_insert.append(row)

    # Insert data - Zilliz will auto-vectorize via the Function
    if data_to_insert:
        print(f"   📊 Inserting {len(data_to_insert)} text chunks...")
        print(f"   ⏳ Zilliz is auto-vectorizing using BAAI/bge-base-en-v1.5...")
        try:
            result = client.insert(
                collection_name=ZILLIZ_COLLECTION,
                data=data_to_insert,
            )
            inserted_count = result.get('insert_count', 0) if isinstance(result, dict) else len(data_to_insert)
            print(f"   ✅ Successfully inserted {inserted_count} chunks")
            print(f"   ✅ Embeddings generated server-side by Zilliz")
        except Exception as e:
            print(f"   ❌ Error inserting data: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("   ⚠️  No data to insert")
        return False

    print()
    return True

# ─────────────────────────────────────────────────────────────────────────────
# Main execution
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    """Run the complete ingestion pipeline."""
    print("=" * 80)
    print("LlamaIndex + Zilliz Cloud Ingestion Pipeline")
    print("=" * 80)
    print()

    try:
        # Step 1: Load documents with LlamaParse
        documents = await load_documents()
        if not documents:
            print("❌ No documents loaded. Exiting.")
            return False

        # Step 2: Process documents
        nodes = create_and_run_pipeline(documents)
        if not nodes:
            print("❌ No nodes produced. Exiting.")
            return False

        # Step 3: Ingest into Zilliz
        success = ingest_into_zilliz(nodes)

        if success:
            print("=" * 80)
            print("✅ INGESTION COMPLETE")
            print("=" * 80)
            print()
            print("Next steps:")
            print("1. Verify in Zilliz Cloud:")
            print(f"   - Collection: {ZILLIZ_COLLECTION}")
            print(f"   - Row count should match the chunks ingested")
            print()
            print("2. Start the backend:")
            print("   cd ~/code/ai-text-opt/nu-finance/backend")
            print("   npm run dev")
            print()
            print("3. Start the frontend:")
            print("   cd ~/code/ai-text-opt/nu-finance/frontend")
            print("   npm run dev")
            print()
            return True
        else:
            print("❌ Ingestion failed")
            return False

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
