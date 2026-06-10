# vector_store.py - ChromaDB integration

import os
import json
from typing import List, Dict, Optional

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def get_collection():
    """Get or create the ChromaDB collection for research findings."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(
            name="research_findings",
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    except Exception as e:
        print(f"[ChromaDB] Error initializing: {e}")
        return None


def store_finding(collection, finding_id: str, text: str, metadata: Dict):
    """Store a research finding in ChromaDB."""
    if collection is None:
        return
    try:
        collection.upsert(
            ids=[finding_id],
            documents=[text],
            metadatas=[metadata]
        )
    except Exception as e:
        print(f"[ChromaDB] Store error: {e}")


def search_findings(collection, query: str, n_results: int = 5) -> List[Dict]:
    """Search stored findings by semantic similarity."""
    if collection is None:
        return []
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        items = []
        if results and results.get("documents"):
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                items.append({"text": doc, "metadata": meta})
        return items
    except Exception as e:
        print(f"[ChromaDB] Search error: {e}")
        return []


def clear_collection(collection):
    """Clear all stored findings."""
    if collection is None:
        return
    try:
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
    except Exception as e:
        print(f"[ChromaDB] Clear error: {e}")
