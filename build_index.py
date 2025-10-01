# import json
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings

# def load_json_text(path):
#     with open(path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     return json.dumps(data, separators=(',', ':'))

# json_files = ["ufdr_report_1.json", "ufdr_report_2.json", "ufdr_report_3.json"]
# documents = []

# for json_file in json_files:
#     json_text = load_json_text(json_file)
#     doc = Document(page_content=json_text, metadata={"source": json_file})
#     documents.append(doc)

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# docs = splitter.split_documents(documents)

# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.from_documents(docs, embedding_model)
# vectorstore.save_local("ufdr_faiss_index")

# print("✅ Combined FAISS index built and saved for all ufdr files!")
import os
import json
import time
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_json_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_metadata(json_data):
    metadata = {}
    def recursive_extract(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                recursive_extract(v, prefix=f"{prefix}{k}_")
            else:
                if isinstance(v, (str, int, float)):
                    metadata[f"{prefix}{k}"] = str(v)
    recursive_extract(json_data)
    return metadata

json_files = ["ufdr_report_1.json", "ufdr_report_2.json", "ufdr_report_3.json"]

documents = []

print("Loading and processing JSON files...")
for json_file in json_files:
    json_data = load_json_text(json_file)
    json_text = json.dumps(json_data, separators=(',', ':'))

    metadata = extract_metadata(json_data)
    metadata["source_file"] = json_file

    doc = Document(page_content=json_text, metadata=metadata)
    documents.append(doc)
print(f"Loaded {len(documents)} documents.")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
print("Splitting documents into chunks...")
docs = []
chunk_count = 0
for doc in splitter.split_documents(documents):
    docs.append(doc)
    chunk_count += 1
    if chunk_count % 100 == 0:
        print(f"Processed {chunk_count} chunks...")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Begin embedding and FAISS index creation...")
start_time = time.time()
vectorstore = FAISS.from_documents(docs, embedding_model)
end_time = time.time()
elapsed = end_time - start_time
print(f"FAISS index build done in {elapsed:.2f} seconds.")

index_path = "ufdr_faiss_combined_index"
vectorstore.save_local(index_path)
print(f"✅ FAISS index built and saved at {index_path}")
