import os
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import csv
import torch
import json
import numpy as np


def get_embeddings(sentences):
    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
    embeddings = model.encode(sentences)
    return embeddings


def convert_markdown_to_chunks(md_file, chunk_size=8000, overlap=100):
    separators = ["\\n\\n", "\\n", " "]
    splitter = RecursiveCharacterTextSplitter(separators, chunk_size, overlap)
    with open(md_file, "r") as f:
        text = f.read()
        # cut off references from paper markdown
        # Reference pattern is: "## References"
        # Cut the reference line and all the following lines
        ref_start = text.find("## References")
        if ref_start != -1:
            text = text[:ref_start]
    chunks = splitter.split_text(text)
    return chunks


def convert_embeddings(source_path, index_path, chunk_size=8000, overlap=100):
    # Search for markdown files in source_path recursively
    markdown_files = []
    for root, dirs, files in os.walk(source_path):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))
    print(markdown_files)

    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)

    if not os.path.exists(index_path):
        os.makedirs(index_path)

    for md_file in markdown_files:
        chunks = convert_markdown_to_chunks(md_file, chunk_size, overlap)
        pure_filename = os.path.splitext(os.path.basename(md_file))[0]
        chunk_filename = os.path.join(index_path, pure_filename + "_chunks.csv")
        vector_filename = os.path.join(index_path, pure_filename + "_vectors.npy")
        embeddings = []
        for chunk_id in range(0, len((chunks)), 8):
            embeddings.extend(model.encode(chunks[chunk_id : chunk_id + 8]))
        with open(chunk_filename, "w", newline="") as csvfile:
            fieldnames = ["id", "chunk"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            chunk_format = lambda string: repr(string)

            for i in range(len(chunks)):
                writer.writerow({"id": i, "chunk": chunk_format(chunks[i])})
        np.save(vector_filename, embeddings)

        # store chunks into
        torch.cuda.empty_cache()
