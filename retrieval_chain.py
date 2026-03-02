from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def retrieve_context(query: str) -> str:
    """Retrieve relevant documents and return combined context."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)