import streamlit as st
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_PATH = os.path.join(BASE_DIR, "faiss_index")

load_dotenv()

st.set_page_config(page_title="Pokmon TCG Pocket Chatbot", page_icon="", layout="centered")

st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stChatFloatingInputContainer {
        bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title(" Pokmon TCG Pocket RAG Chatbot")
st.markdown("Ask me anything about the **Genetic Apex** and other Pocket series cards!")

with st.sidebar:
    st.header("Pokmon TCG Pocket")
    st.info("Built with LangChain, OpenAI, and FAISS.")
    
    st.subheader("📚 Library Stats")
    st.write("- **Sets Loaded:** 15 (Series: `tcgp`)")
    st.write("- **Total Cards:** ~2,400 (Unique)")
    st.write("- **RAG Method:** Ultimate-Context Search (k=100)")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Load Resources (Cached)
@st.cache_resource
def load_rag_pipeline():
    # Dynamic/Lazy Imports to keep the initial UI load instant
    from langchain_openai import ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if not os.path.exists("faiss_index"):
        return None, None
    
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 100, "fetch_k": 200}
    )
    
    # Initialize LLM (OpenAI GPT-4o-mini)
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        openai_api_key=os.environ.get("OPENAI_KEY")
    )
    
    # System Prompt
    system_prompt = (
        "You are an expert Pokemon TCG Pocket assistant. "
        "Use the provided database context to answer the user's question accurately. "
        "IMPORTANT: If the 'GLOBAL LIBRARY OVERVIEW' in the context provides specific counts (like the number of sets or Mega Evolutions), "
        "trust those numbers as the absolute truth for the library. "
        "Then, list every unique Pokemon name that matches the query found across the entire context to support your answer.\n\n"
        "DATABASE CONTEXT:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create Chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever

# Chat Input Block
if user_input := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
        
    with st.chat_message("assistant"):
        # Lazy Loading the Pipeline on the first interaction
        if "rag_pipeline" not in st.session_state:
            with st.status("Initializing Pokedex Brain (first time only)...", expanded=True) as status:
                st.write("📂 Loading heavy AI libraries...")
                # Dynamic/Lazy Imports
                from langchain_openai import ChatOpenAI
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_core.runnables import RunnablePassthrough
                from langchain_core.output_parsers import StrOutputParser
                
                st.write("🧠 Loading Embedding Model (MiniLM)...")
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                st.write("📚 Indexing 2,400+ cards from FAISS...")
                if not os.path.exists(FAISS_PATH):
                     st.error("Error: Vector index not found! Please run 'python ingest.py' first.")
                     st.stop()
                vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 30, "fetch_k": 90})
                
                st.write("🌩️ Connecting to OpenAI (GPT-4o-mini)...")
                llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", openai_api_key=os.environ.get("OPENAI_KEY"))
                
                # Setup Pipeline (Strict Verification)
                system_prompt = (
                    "You are an expert Pokemon TCG Pocket assistant. "
                    "Use ONLY the provided database context below to answer questions. "
                    "CRITICAL RULES:\n"
                    "1. RARITY VERIFICATION: When a user asks about a specific rarity (e.g., '2 Diamond'), "
                    "you MUST check the 'Rarity' field for EACH card before listing it. "
                    "Do NOT guess — if a card's Rarity field says 'One Diamond', do NOT include it in a 'Two Diamond' answer.\n"
                    "2. TYPE VERIFICATION: Check the actual 'Types' field on each card. "
                    "Do not assume a Pokemon's type from general knowledge.\n"
                    "3. GLOBAL COUNTS: Trust the 'GLOBAL LIBRARY OVERVIEW' for total counts.\n"
                    "4. COUNTING RULE: After making your list, COUNT the items you listed. "
                    "The number you state MUST match the actual number of items in your list. "
                    "If the GLOBAL LIBRARY OVERVIEW has a total count, state that total and clarify "
                    "'here are X from the database context' if you can only show a partial list.\n"
                    "5. If a card's metadata does not match the user's query, exclude it.\n\n"
                    "DATABASE CONTEXT:\n{context}"
                )
                prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                
                st.session_state.rag_pipeline = (
                    {"context": retriever | format_docs, "input": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                st.session_state.retriever = retriever
                status.update(label="Brain Initialized!", state="complete", expanded=False)

        rag_pipeline = st.session_state.rag_pipeline
        retriever = st.session_state.retriever
        
        with st.spinner("Searching through 15 sets..."):
            try:
                # Get response
                answer = rag_pipeline.invoke(user_input)
                st.markdown(answer)
                
                # Manual retrieval for images
                docs = retriever.invoke(user_input)
                if docs:
                    with st.expander("View Reference Card Images"):
                        # Smart Filter: Strict 1-to-1 mapping by name
                        # Sort by longest name first to prevent substring false matches
                        sorted_docs = sorted(docs, key=lambda d: len(d.metadata.get("name", "")), reverse=True)
                        mentioned_docs = []
                        seen_names = set()
                        lower_answer = answer.lower()
                        
                        for doc in sorted_docs:
                            name = doc.metadata.get("name")
                            if not name:
                                continue
                            lower_name = name.lower()
                            already_covered = any(lower_name in matched for matched in seen_names)
                            if already_covered:
                                continue
                            if lower_name in lower_answer:
                                mentioned_docs.append(doc)
                                seen_names.add(lower_name)
                                
                        if mentioned_docs:
                            st.write(f"Showing {len(mentioned_docs)} images to match your list:")
                            cols = st.columns(min(len(mentioned_docs), 3))
                            for i, doc in enumerate(mentioned_docs):
                                img_url = doc.metadata.get("image")
                                card_name = doc.metadata.get("name")
                                with cols[i % 3]:
                                    if img_url:
                                        st.image(img_url, caption=card_name)
                                    else:
                                        st.info(f"🎨 No artwork URL found for {card_name}")
                        else:
                            st.write("No specific card art matches the conversation context.")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = str(e)
                if "rate_limit" in error_msg.lower() or "429" in error_msg or "413" in error_msg:
                    st.error("⚡ **Rate limit reached!** The AI model has hit its token limit. Please wait about 60 seconds and try again.")
                    st.info("💡 **Tip:** Shorter questions use fewer tokens and are less likely to hit the limit.")
                elif "model_decommissioned" in error_msg.lower():
                    st.error("🚫 **Model unavailable.** The AI model has been retired. Please contact the developer to update the model.")
                else:
                    st.error(f"❌ **Something went wrong:** {error_msg}")
                    st.info("💡 Try refreshing the page or waiting a moment before trying again.")

