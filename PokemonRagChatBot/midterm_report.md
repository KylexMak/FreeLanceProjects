# Pokemon TCG Pocket RAG Chatbot: Midterm Project Report

## 1. Introduction
The Pokemon Trading Card Game (TCG) Pocket is a recent digital adaptation of the classic card game. With over 2,400 unique cards across multiple sets, players frequently need a way to query specific card statistics, abilities, and rarity distributions. This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that provides accurate, metadata-verified answers by combining **LangChain**, **FAISS**, and the **OpenAI API**.

## 2. Technical Implementation
The system is divided into two main components: the Data Ingestion Pipeline and the Streamlit Chat Interface.

### 2.1 Data Ingestion (`ingest.py`)
We used the **TCGdex API** to fetch comprehensive card data. The data is converted into LangChain `Document` objects with structured text representations.

```python
def process_card_to_doc(card):
    # Extracts metadata and creates a searchable string
    name = card.get("name", "Unknown")
    hp = card.get("hp", "N/A")
    types = ", ".join(card.get("types", []))
    text_content = f"Card Name: {name}\nHP: {hp}\nTypes: {types}\n..."
    
    metadata = {
        "id": card.get("id"),
        "name": name,
        "image": f"{card.get('image')}/high.webp"
    }
    return Document(page_content=text_content, metadata=metadata)
```

### 2.2 RAG Chain Architecture (`app.py`)
The chatbot uses the **GPT-4o-mini** model from OpenAI for high-speed inference. The retrieval step uses **FAISS** with **MiniLM-L6-v2** embeddings.

```python
# Create Chain using LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)
```

## 3. Challenges and Solutions

### 3.1 Mitigation of LLM Hallucinations (Types and Rarity)
One of the most significant challenges was the LLM hallucinating card types based on general Pokemon knowledge. For example, it might assume a "Pikachu" is always an Electric type, even if the specific card data says otherwise (e.g., promotional variants).

**Solution:** We implemented a "Strict Verification" system prompt that forbids the model from using external knowledge.

```python
system_prompt = (
    "Use ONLY the provided database context below to answer questions.\n"
    "CRITICAL RULES:\n"
    "1. TYPE/RARITY VERIFICATION: Verify all stats directly from card data.\n"
    "2. DATASET SCOPE: You are permitted to answer 'meta' questions "
    "about your dataset (sets, totals) using GLOBAL LIBRARY GROUND TRUTH.\n"
    "3. OFF-TOPIC RULE: Only refuse questions completely unrelated to Pokemon."
)
```

### 3.2 Quantitative Accuracy: The Global Library Overview
When asked "How many sets are in the game?", the RAG system would often fail because the retriever only returns the "Top-K" most similar cards, not the entire database.

**Solution:** We injected a "Global Library Overview" document into the FAISS index that contains pre-calculated statistics about the entire collection.

```python
# Global Overview document to handle high-level questions
overview_text = (
    "GLOBAL LIBRARY OVERVIEW: This library contains 15 sets... "
    "Total cards: approx 2,480. Rarity breakdown: One Diamond: 448 unique cards..."
)
overview_doc = Document(page_content=overview_text, metadata={"name": "Global Overview"})
docs.append(overview_doc)
```

### 3.3 Visual Synchronization (Image Gallery)
A common issue in RAG chatbots is the "Stale Reference" problem: where the AI mentions a card in text but the UI shows a different set of cards retrieved by the search.

**Solution:** We implemented a "Smart Filter" that cross-references the AI's final natural language response with the retrieved card metadata to ensure only mentioned cards are displayed in the image gallery.

```python
# Filter images to only those mentioned in the AI's final answer
mentioned_docs = []
for doc in docs:
    if doc.metadata.get("name").lower() in answer.lower():
        mentioned_docs.append(doc)

# Render images in Streamlit columns
cols = st.columns(3)
for i, doc in enumerate(mentioned_docs):
    cols[i % 3].image(doc.metadata.get("image"), caption=doc.metadata.get("name"))
```

### 3.4 User Experience: Lazy Loading heavy AI Libraries
During initial testing, the application suffered from a "Blank Screen" problem. Because LangChain, HuggingFace, and FAISS are heavy libraries, the Streamlit app would take 10-15 seconds to load before the user could even see the chat interface.

**Solution:** We implemented **Lazy Loading** by moving the library imports and model initialization inside the first user interaction loop. This allows the UI to render instantly, providing the user with a "Starting..." status indicator only when they first ask a question.

```python
# Lazy Loading the Pipeline inside the first chat interaction
if "rag_pipeline" not in st.session_state:
    with st.status("Initializing Pokedex Brain (first time only)..."):
        # Imports happen only when needed
        from langchain_openai import ChatOpenAI
        ...
        st.session_state.rag_pipeline = create_chain()
```

### 3.5 Information Enrichment: Evolution Chain Mapping
A common user query involves the evolution status of a Pokémon (e.g., "What does this card evolve from?"). While base RAG can find card names, it often misses relational data if not explicitly indexed.

**Solution:** We updated the `ingest.py` script to pull the `stage` and `evolveFrom` fields from the TCGdex API and include them directly in the searchable document text.

```python
# Updated ingestion logic to capture evolution relations
stage = card.get("stage", "Basic")
evolve_from = card.get("evolveFrom")
text_parts.append(f"Stage: {stage}")
if evolve_from:
    text_parts.append(f"Evolves From: {evolve_from}")
```

## 4. Performance Results
By leveraging **OpenAI's high-speed inference**, the average response time for a complex query (retrieving 30+ documents and generating a response) remains highly competitive (under 2 seconds). The use of **FAISS** allows for local vector storage, eliminating the need for a costly cloud-based vector database.

## 5. Conclusion
This project demonstrates that a robust RAG architecture can transform a static API into an intelligent, interactive assistant. By prioritizing strict prompt engineering and pre-calculating library-wide metadata, we successfully addressed common LLM failures regarding factual accuracy and visual synchronization.
