import sys
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm.auto import tqdm

# Ensure project root is added to Python path
project_root=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
if project_root not in sys.path:
  sys.path.append(project_root)

# Absolute path of repositories
DATA_DIR="/content/ai-foundations-rag-lab/data"
OUTPUT_DIR="/content/ai-foundations-rag-lab/outputs"

INDEX_PATH=os.path.join(OUTPUT_DIR,"faiss_index.index")
META_PATH=os.path.join(OUTPUT_DIR,"metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"

_model=None

def _get_model():
  """Load or return cached Sentence Transformer model"""

  global _model
  if _model is None:
    print(f"Loading Sentence Transformer: {MODEL_NAME}")
    _model=SentenceTransformer(MODEL_NAME)
  return _model

def load_index_and_metadata(index_path=INDEX_PATH,meta_path=META_PATH):
  # File exists checker

  if not os.path.exists(index_path):
    raise FileNotFoundError("Unable to load index file.")
  
  if not os.path.exists(meta_path):
    raise FileNotFoundError("Unable to locate metadata file.")
  
  print(f"Loading index file from - {index_path}")
  index=faiss.read_index(index_path)

  print(f"Loading metadata file from - {meta_path}")

  with open(meta_path,"r",encoding="utf-8")as f:
    metadata=json.load(f)

  print("Index and Metadata has been loaded successfully.")
  
  return index,metadata


def embed_query(query:str,model):
  # Embed the query and normalize it for cosine similarity

  if not isinstance(query,str):
    raise ValueError("Query must be string")

  q_vec=model.encode([query],convert_to_numpy=True).astype(np.float32)
  faiss.normalize_L2(q_vec)
  return q_vec

def retrieve(
  query:str,
  top_k:int=5,
  index_path=INDEX_PATH,
  meta_path=META_PATH,
  model_name=MODEL_NAME
):

  # Main retrieval function - Embed - > Search -> Retrieve
  index,metadata=load_index_and_metadata(index_path,meta_path)
  model=_get_model()

  q_emb=embed_query(query,model)

  # FAISS Search
  distance,indices = index.search(q_emb,top_k)

  results=[]

  for score,idx in zip(distance[0],indices[0]):
    if idx<0 or idx>=len(metadata):
      continue
    meta=metadata[idx]

    results.append({
      "score":float(score),
      "doc_id":meta["doc_id"],
      "passage_id":meta["passage_id"],
      "text":meta["text"]
    })

  return results


if __name__=="__main__":
  q="how have hybrid engines changed Formula 1 strategy?"
  q_res=retrieve(q,top_k=5)

  for i,j in enumerate(q_res,1):
    print(f"Query Score {i}: {j['score']:.4f}")
    print(f"Doc: {j['doc_id']}")
    print(f"Passage Id: {j['passage_id']}")
    print(f"Text: {j['text'][:300]}")




