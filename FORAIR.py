#!/usr/bin/env python3
# FORAIR - Forensic RAG AI CLI Tool
# © 2025 Shane D. Shook, All Rights Reserved

import os
import glob
import json
import argparse
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from colorama import Fore, Style, init

init(autoreset=True)

# Configuration
CSV_DIR = "D:/extracts/"
CHUNK_SIZE = 500
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INDEX_FILE = "csv_chunks.index"
META_FILE = "csv_chunks.pkl"
LLAMA_MODEL_PATH = "TinyLLaMA-1.1B-Chat.Q4_K_M.gguf"
N_CTX = 2048

def load_and_chunk_csvs(csv_dir):
    chunks = []
    metadata = []
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    for filepath in csv_files:
        artifact_type = os.path.splitext(os.path.basename(filepath))[0]
        try:
            df = pd.read_csv(filepath, dtype=str, encoding='utf-8', errors='ignore')
            for i, row in df.iterrows():
                text = row.dropna().astype(str).str.cat(sep=" | ")
                if len(text.strip()) == 0:
                    continue
                chunks.append(text[:CHUNK_SIZE])
                metadata.append({
                    "artifact": artifact_type,
                    "file": os.path.basename(filepath),
                    "row": i
                })
        except Exception as e:
            print(f"{Fore.RED}Error reading {filepath}: {e}")
    return chunks, metadata

def build_faiss_index(chunks, metadata, model):
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "metadata": metadata}, f)
    print(f"{Fore.GREEN}✅ Index built and saved.")

def query_index(question, model, top_k=5):
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    metadata = data["metadata"]
    q_emb = model.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    results = [(chunks[i], metadata[i]) for i in I[0]]
    return results

def build_prompt(question, retrieved_chunks):
    context = "\n\n".join(
        [f"[{meta['artifact']} row {meta['row']} in {meta['file']}]: {text}" 
         for text, meta in retrieved_chunks]
    )
    prompt = f"""You are a forensic analysis assistant using evidence from multiple sources.

Context:
{context}

Question: {question}
Answer:"""
    return prompt

def run_llama(prompt, model_path, n_ctx):
    llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False)
    output = llm(prompt, max_tokens=256, stop=["</s>", "\n\n"])
    return output["choices"][0]["text"].strip()

def main():
    parser = argparse.ArgumentParser(description="FORAI - Forensic RAG CLI Tool")
    parser.add_argument("--build", action="store_true", help="Build the FAISS index from forensic CSVs")
    parser.add_argument("--ask", type=str, help="Ask a forensic question")
    parser.add_argument("--json-output", type=str, help="Save the answer and sources to a JSON file")
    args = parser.parse_args()

    model = SentenceTransformer(EMBEDDING_MODEL)

    if args.build:
        print(f"{Fore.YELLOW}🔄 Building FAISS index from {CSV_DIR} ...")
        chunks, metadata = load_and_chunk_csvs(CSV_DIR)
        build_faiss_index(chunks, metadata, model)

    elif args.ask:
        if not os.path.exists(INDEX_FILE):
            print(f"{Fore.RED}❌ Index not found. Run with --build first.")
            return
        print(f"{Fore.CYAN}🔍 Asking: {args.ask}")
        retrieved = query_index(args.ask, model)
        prompt = build_prompt(args.ask, retrieved)
        print(f"{Fore.MAGENTA}🧠 Prompting TinyLLaMA...\n")
        answer = run_llama(prompt, LLAMA_MODEL_PATH, N_CTX)
        print(f"{Fore.GREEN}Answer:\n{Style.BRIGHT}{answer}")
        if args.json_output:
            output = {
                "question": args.ask,
                "answer": answer,
                "sources": retrieved
            }
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            print(f"{Fore.BLUE}📁 Output written to {args.json_output}")

if __name__ == "__main__":
    main()
