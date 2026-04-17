import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def test():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Testing for Raichu ex
    query = "Raichu ex"
    docs = vector_store.similarity_search(query, k=5)
    
    print(f"Query: {query}")
    for i, doc in enumerate(docs):
        print(f"\n--- Result {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content}")

if __name__ == "__main__":
    test()
