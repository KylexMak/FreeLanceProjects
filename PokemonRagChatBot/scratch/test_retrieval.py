import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def test():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists("faiss_index"):
        print("Index not found!")
        return
    
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    query = "how many sets of cards are you trained on"
    docs = vector_store.similarity_search(query, k=5)
    
    print(f"Query: {query}")
    for i, doc in enumerate(docs):
        print(f"\n--- Result {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content snippet: {doc.page_content[:200]}...")

if __name__ == "__main__":
    test()
