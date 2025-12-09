import sys, os

# Ensure project root is on path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from src.ingest import build_corpus
from tqdm import tqdm

"""
Convert passages into dense vectors for FAISS (Google Colab Compatible)
"""

MODEL_NAME = "all-MiniLM-L6-v2"

# Absolute paths for Colab
DATA_DIR = "/content/ai-foundations-rag-lab/data"
OUTPUT_DIR = "/content/ai-foundations-rag-lab/outputs"
INDEX_PATH = f"{OUTPUT_DIR}/faiss_index.index"
META_PATH = f"{OUTPUT_DIR}/metadata.json"


def embed_corpus(corpus, model):

    if not corpus:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    print("\n[1/4] Embedding passages...")
    texts = [item["text"] for item in corpus]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings


def build_faiss_index(embeddings, dim, index_path=INDEX_PATH):

    # when vectors are unit-normalized, inner product = cosine similarity
    print("\n[2/4] Normalizing embeddings (L2)...")
    faiss.normalize_L2(embeddings)

    print("[3/4] Building FAISS IndexFlatIP and adding vectors...")
    index = faiss.IndexFlatIP(dim)

    batch_size = 500
    for i in tqdm(range(0, len(embeddings), batch_size)):
        batch = embeddings[i:i + batch_size]
        index.add(batch)

    print(f"Saving FAISS index to disk at: {index_path}")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    return index


def save_metadata(corpus, meta_path=META_PATH):

    print(f"\n[4/4] Saving metadata.json to: {meta_path}")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)


def main():

    print("Loading corpus from:", DATA_DIR)
    corpus = build_corpus(data_dir=DATA_DIR)
    print(f"Loaded {len(corpus)} passages.")

    print("\nLoading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    embeddings = embed_corpus(corpus, model).astype(np.float32)
    dim = embeddings.shape[1]

    build_faiss_index(embeddings, dim)
    save_metadata(corpus)

    print("\nIndex and metadata saved successfully!")


if __name__ == "__main__":
    main()
