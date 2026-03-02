from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import json

# Use the same embedding model as preprocessing.py and rag.py
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Load knowledge base
with open("./data/knowledge_base.json", "r", encoding="utf-8") as f:
    kb_data = json.load(f)

# Convert to LangChain Documents
documents = [
    Document(
        page_content=item["content"],
        metadata={"id": item["id"], "title": item["title"]}
    )
    for item in kb_data
]

print(f"Building FAISS index from {len(documents)} documents...")

# Build and save FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("faiss_index")

print("FAISS index saved to ./faiss_index/")
