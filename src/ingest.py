import os
import re
import json

"""
Ingest + Splitter
Purpose - Large docs must be chunked so the retrieval return precise snippet to quote as citation.
"""

def load_markdown_files(data_dir):
    docs=[]

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".md"):
            continue
        path=os.path.join(data_dir,fname)

        with open(path,"r",encoding="utf-8") as f:
            text=f.read()
        
        docs.append({"id":fname, "text":text,"path":path})

    return docs

def split_into_passages(text,max_words=600,overlap=30):
    # simple word based sliding window splitter

    words=re.split(r"\s+",text.strip())
    passages=[]
    start=0
    n=len(words)

    while start<n:
        end=min(start+max_words,n)
        passage=" ".join(words[start:end])
        passages.append(passage)

        if end == n:
            break
        else:
            start=end-overlap
    return passages

def build_corpus(data_dir="/content/ai-foundations-rag-lab/data"):
    docs=load_markdown_files(data_dir)
    items=[]

    for doc in docs:
        passages=split_into_passages(doc["text"])

        for i,p in enumerate(passages):
            items.append({
                "doc_id":doc["id"],
                "passage_id":f"{doc['id']}_p{i}",
                "text":p
            })
    return items

if __name__=="__main__":
    corpus=build_corpus()
    print(f"Built corpus for your data. {len(corpus)} passages from your markdown.")
