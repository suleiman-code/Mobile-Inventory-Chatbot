import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import re

# RAG Imports
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
MODEL_NAME = "gpt-3.5-turbo"
CSV_FILE = "MobileInventory.csv"

# Initialize OpenAI Client (Base OpenAI used for Final Chat completion)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("OPENAI_API_KEY not found. Please set it as an environment variable or in .streamlit/secrets.toml")
        st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="SLM Mobile AI - RAG Agent", layout="wide", page_icon="📱")

# Custom Premium Styling
st.markdown("""
    <style>
    :root {
        --bg-color: #ffffff;
        --sidebar-bg: #f8f9fa;
        --text-main: #1a1a1a;
        --accent-blue: #00d2ff;
        --border-color: #e0e0e0;
    }
    .main { background-color: var(--bg-color); color: var(--text-main); }
    [data-testid="stSidebar"] { background-color: var(--sidebar-bg); border-right: 1px solid var(--border-color); }
    .stChatMessage { border-bottom: 1px solid #f0f0f0 !important; padding: 1.5rem 0 !important; max-width: 800px; margin: 0 auto !important; }
    .logo-container { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding: 10px; }
    .logo-text { font-size: 1.4rem; font-weight: 700; color: #1a1a1a; }
    </style>
""", unsafe_allow_html=True)

# --- RAG INITIALIZATION ---
@st.cache_resource
def initialize_rag():
    try:
        # 1. Document Loader
        loader = CSVLoader(file_path=CSV_FILE, encoding='utf-8')
        documents = loader.load()

        # 2. Chunking (Each row is a chunk for structured data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(documents)

        # 3. Embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        # 4. Vector DB (FAISS)
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        return vector_db
    except Exception as e:
        st.error(f"Error initializing RAG: {e}")
        return None

vector_db = initialize_rag()

def rerank_context(query, initial_docs):
    """
    Reranking Concept: 
    After initial retrieval from FAISS (which uses semantic similarity), 
    we use the LLM to 'rerank' or filter the most relevant items 
    to ensure the context window is high-quality.
    """
    if not initial_docs:
        return "No data found."
    
    # Combine initial docs into a list for the LLM to evaluate
    context_to_rerank = "\n\n".join([f"DOC {i+1}:\n{doc.page_content}" for i, doc in enumerate(initial_docs)])
    
    rerank_prompt = f"""You are a RAG Re-ranker. Given the query '{query}', select the TOP 8 most relevant mobile phone models from the list below. 
    Return only the content of those models, separated by newlines.
    
    MODELS FOUND:
    {context_to_rerank}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a professional inventory filterer."},
                      {"role": "user", "content": rerank_prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except:
        # Fallback to initial docs if reranking fails
        return "\n\n".join([doc.page_content for doc in initial_docs[:8]])

def get_relevant_context(query):
    if vector_db:
        # Stage 1: Retrieval (Get 20 candidates)
        initial_docs = vector_db.similarity_search(query, k=20)
        
        # Stage 2: Reranking (Refine to best 8)
        reranked_docs = rerank_context(query, initial_docs)
        return reranked_docs
    return "No inventory data found."

# --- DATA FOR DASHBOARD ---
def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return None

df = load_data()

with st.sidebar:
    st.markdown("""
        <div class="logo-container">
            <img src="https://img.icons8.com/ios-filled/50/4a90e2/iphone-x.png" width="35">
            <span class="logo-text">SLM Mobile AI</span>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        st.subheader("Inventory Metrics")
        m1, m2 = st.columns(2)
        with m1:
            top_10 = ['Apple', 'Samsung', 'Google', 'OnePlus', 'Xiaomi', 'Vivo', 'Oppo', 'Realme', 'Infinix', 'Tecno']
            display_df = df[df['Brand'].isin(top_10)]
            st.metric("Models Available", len(display_df))
        with m2:
            st.metric("Total Brands", display_df['Brand'].nunique())
        st.divider()
        st.info("✨ Providing expert mobile advice and current availability.")
    else:
        st.error(f"Inventory data not accessible.")

# Chat initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I'm your SLM Mobile expert. How can I help you find the perfect device today?"}]

# Header
st.markdown("""
    <div style="text-align: center; padding: 20px 0 40px 0;">
        <h1 style="font-size: 3rem; font-weight: 800; color: #1a1a1a; margin-bottom: 5px;">
            📱 SLM <span style="color: #00d2ff;">Mobile</span> Expert
        </h1>
        <p style="color: #666; font-size: 1.1rem; letter-spacing: 1px;">PREMIUM ASSISTANCE</p>
    </div>
""", unsafe_allow_html=True)

chat_container = st.container()

with chat_container:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

if prompt := st.chat_input("Message SLM Mobile AI..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Global Awareness (To handle general questions like 'what brands do you have?')
    all_brands = sorted(df['Brand'].unique().tolist()) if df is not None else []
    total_inventory = len(df) if df is not None else 0
    global_inventory_summary = f"Total Unique Models: {total_inventory}. Brands in Stock: {', '.join(all_brands)}."

    # 2. Internal RAG Logic (Hidden from user)
    initial_docs = vector_db.similarity_search(prompt, k=20) if vector_db else []
    context_data = rerank_context(prompt, initial_docs) if initial_docs else "No specific context found."

    # 3. Construct Professional Prompt
    system_prompt = f"""You are SLM Mobile AI, a highly sophisticated and professional mobile expert assistant.
    
    **GLOBAL KNOWLEDGE:**
    {global_inventory_summary}
    
    **SPECIFIC INVENTORY DATA (FOR RELEVANT QUERIES):**
    {context_data}
    
    **INSTRUCTIONS:**
    1. **Conversational Excellence**: Respond naturally. If asked about brands or general stock, use the GLOBAL KNOWLEDGE. If asked about specific models or specs, use the SPECIFIC INVENTORY DATA.
    2. **Accuracy**: Do not mention technical terms like "RAG," "Vector DB," or "retrieved data." 
    3. **Presentation**: Use bold text for model names. Use structured tables for comparing 3 or more devices.
    4. **Pricing**: Always mention prices in PKR.
    5. **Integrity**: If a user asks for something we don't carry, check the GLOBAL KNOWLEDGE brands and suggest the best alternative from those brands using the SEARCH DATA.
    6. **Tone**: Helpful, elite, and precise. Like a premium concierge.
    """

    # 4. Call Model with a clean loading state
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            messages_payload = [{'role': 'system', 'content': system_prompt}] + st.session_state["messages"]
            
            with st.spinner(" "): 
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages_payload,
                    stream=False,
                )
                full_response = response.choices[0].message.content
                
            response_placeholder.markdown(full_response)
            st.session_state["messages"].append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error("I encountered a brief connection issue. Please try again.")
