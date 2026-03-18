import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import re
import time

# RAG Imports
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Configuration
MODEL_NAME = "gpt-3.5-turbo"
CSV_FILE = "MobileInventory.csv"

# Initialize OpenAI Client (Base OpenAI used for Final Chat completion)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError()
    except Exception:
        st.error("OPENAI_API_KEY not found. Please set it as an environment variable or in .streamlit/secrets.toml")
        st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="SLM Mobile AI", layout="wide", page_icon="https://img.icons8.com/ios-filled/50/4a90e2/iphone-x.png")

# Custom Premium Styling - ChatGPT Style
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main Background */
    .stApp, .main { 
        background-color: #ffffff; 
        color: #0d0d0d;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #f9f9f9; 
        border-right: 1px solid #e5e5e5;
    }
    
    /* Input Area Container */
    [data-testid="stChatInput"] {
        border-radius: 24px !important;
        background-color: #f4f4f4 !important;
        border: 1px solid #e5e5e5 !important;
        padding: 4px 6px !important;
        box-shadow: none !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stChatInput"]:focus-within {
        background-color: #f4f4f4 !important;
        border: 1px solid #b4b4b4 !important;
        box-shadow: 0 0 0 2px rgba(0,0,0,0.05) !important;
    }
    
    /* Remove red inner border on focus */
    [data-testid="stChatInput"] > div {
        border: none !important;
        box-shadow: none !important;
        background-color: transparent !important;
    }
    
    /* Input send button */
    [data-testid="stChatInput"] button {
        background-color: #e5e5e5 !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        outline: none !important;
    }
    
    /* Arrow color default (light) */
    [data-testid="stChatInput"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        width: 18px !important;
        height: 18px !important;
        margin: auto !important;
    }
    
    /* Active State: When there is text */
    [data-testid="stChatInput"] button:not(:disabled) {
        background-color: #000000 !important;
    }
    
    /* Hover on active state */
    [data-testid="stChatInput"] button:not(:disabled):hover {
        background-color: #333333 !important;
        transform: scale(1.1);
    }
    
    /* Chat Messages Animation */
    @keyframes messageEntry {
        0% { opacity: 0; transform: translateY(12px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 1.2rem 0 !important;
        max-width: 48rem;
        margin: 0 auto !important;
        animation: messageEntry 0.5s cubic-bezier(0.19, 1, 0.22, 1) forwards;
    }
    
    /* Avatars override */
    [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] { display: none !important; }
    [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
    
    .user-container { width: 100%; display: flex; justify-content: flex-end; }
    .user-bubble {
        background-color: #f3f3f3; 
        color: #1f1f1f; 
        padding: 10px 18px; 
        border-radius: 18px; 
        max-width: 80%;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .bot-container { display: flex; align-items: flex-start; width: 100%; gap: 16px; }
    .bot-sparkle { 
        width: 28px; 
        height: 28px; 
        margin-top: 4px; 
        flex-shrink: 0;
        transition: transform 0.3s ease;
    }
    .bot-container:hover .bot-sparkle {
        transform: scale(1.1);
    }
    
    .bot-content { 
        flex-grow: 1; 
        color: #131313; 
        padding-top: 4px; 
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Typing Cursor */
    .cursor {
        display: inline-block;
        width: 2px;
        height: 1.1em;
        background-color: #4a90e2;
        margin-left: 2px;
        vertical-align: middle;
        animation: blink 0.8s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }

    /* Pulse for Thinking State */
    .pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }

    .logo-container { display: flex; align-items: center; gap: 12px; margin-bottom: 2rem; padding: 0 10px; }
    .logo-text { font-size: 1.1rem; font-weight: 600; color: #1a1a1a; letter-spacing: -0.01em; }
    
    .block-container { padding-top: 3rem !important; }

    /* Tables Improvement */
    table {
        border-collapse: collapse;
        margin: 15px 0;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #eee;
    }
    th {
        background-color: #f8f9fa;
        text-align: left;
        padding: 12px;
        font-weight: 600;
        border-bottom: 2px solid #eee;
    }
    td {
        padding: 10px 12px;
        border-bottom: 1px solid #eee;
    }
    tr:last-child td { border-bottom: none; }
    </style>
""", unsafe_allow_html=True)

# --- RAG INITIALIZATION ---
@st.cache_resource
def initialize_rag():
    try:
        # 1 & 2. Improved Data Chunking Strategy for Tabular Data
        import pandas as pd
        df = pd.read_csv(CSV_FILE)
        chunks = []
        for index, row in df.iterrows():
            # Convert row to an informative paragraph
            content = f"Model: {row.get('Model', 'N/A')}, Brand: {row.get('Brand', 'N/A')}. "
            specs = [f"{col}: {row[col]}" for col in df.columns if col not in ['Model', 'Brand'] and pd.notna(row.get(col))]
            content += ", ".join(specs)
            doc = Document(page_content=content, metadata={"source": f"row_{index}"})
            chunks.append(doc)

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

def rewrite_query(user_query, chat_history):
    """Rewrite query to include context from chat history."""
    if len(chat_history) <= 2:
        return user_query
    # Get last 6 messages
    recent_history = [msg for msg in chat_history[-6:] if msg['role'] != 'system']
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    prompt = f"Given the conversation history and the latest user query, rewrite the user query to be a standalone search query containing all necessary context (e.g., brand, model mentioned previously) so it can be used to search an inventory database. DO NOT answer it, just rewrite.\n\nHistory:\n{history_text}\n\nLatest Query: {user_query}\n\nStandalone Query:"
    
    try:
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return user_query

# --- DATA FOR DASHBOARD ---
def load_data():
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return None

df = load_data()

with st.sidebar:
    st.markdown("""
        <div class="logo-container">
            <div style="background-color: white; border: 1px solid #e5e5e5; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center;">
                <img src="https://img.icons8.com/ios-filled/50/4a90e2/iphone-x.png" width="20">
            </div>
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
    st.session_state["messages"] = []

# 1. Capture prompt at the top to control UI state
prompt = st.chat_input("Message SLM Mobile AI...")

# 2. Header (Only show if no messages AND no prompt just entered)
if len(st.session_state["messages"]) == 0 and not prompt:
    st.markdown("""
        <div style="text-align: center; padding: 15vh 0 5vh 0;">
            <div style="background-color: white; border: 1px solid #e5e5e5; width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 20px auto; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <img src="https://img.icons8.com/ios-filled/50/4a90e2/iphone-x.png" width="35">
            </div>
            <h1 style="font-size: 2rem; font-weight: 600; color: #0d0d0d; margin-bottom: 10px;">
                How can I help you today?
            </h1>
            <p style="color: #666; font-size: 1rem;">Ask me anything about our premium mobile inventory.</p>
        </div>
    """, unsafe_allow_html=True)

chat_container = st.container()
bot_icon = '''<img class="bot-sparkle" src="https://img.icons8.com/ios-filled/50/4a90e2/iphone-x.png" width="24" height="24">'''

with chat_container:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar=None):
            if msg["role"] == "user":
                st.markdown(f'<div class="user-container"><div class="user-bubble">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-container">{bot_icon}<div class="bot-content">{msg["content"]}</div></div>', unsafe_allow_html=True)

# 3. Handle new prompt
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=None):
        st.markdown(f'<div class="user-container"><div class="user-bubble">{prompt}</div></div>', unsafe_allow_html=True)

    with st.chat_message("assistant", avatar=None):
        response_placeholder = st.empty()
        # Custom thinking state for a smoother transition
        response_placeholder.markdown(f'<div class="bot-container"><img class="bot-sparkle pulse" src="https://img.icons8.com/ios-filled/50/4a90e2/iphone-x.png" width="24" height="24"><div class="bot-content pulse">Thinking...</div></div>', unsafe_allow_html=True)
        
        full_response = ""
        
        try:
            # Global Awareness
            all_brands = sorted(df['Brand'].unique().tolist()) if df is not None else []
            total_inventory = len(df) if df is not None else 0
            global_inventory_summary = f"Total Unique Models: {total_inventory}. Brands in Stock: {', '.join(all_brands)}."

            # RAG Logic
            standalone_query = rewrite_query(prompt, st.session_state["messages"])
            initial_docs = vector_db.similarity_search(standalone_query, k=20) if vector_db else []
            context_data = rerank_context(standalone_query, initial_docs) if initial_docs else "No specific context found."

            # Construct Prompt
            is_first_message = len(st.session_state["messages"]) == 1
            first_msg_guidance = ""
            if is_first_message:
                brands_to_show = all_brands if len(all_brands) <= 15 else all_brands[:15]
                first_msg_guidance = f"IMPORTANT: This is the user's FIRST message. Start with a warm greeting and briefly list the available brands ({', '.join(brands_to_show)}...) to guide their inquiry."

            system_prompt = f"""You are SLM Mobile AI, a highly sophisticated and professional mobile expert assistant.
            
            {first_msg_guidance}

            **GLOBAL KNOWLEDGE:**
            {global_inventory_summary}
            
            **SPECIFIC INVENTORY DATA (FOR RELEVANT QUERIES):**
            {context_data}
            
            **INSTRUCTIONS:**
            1. **Conversational Excellence**: Respond naturally. If asked about brands or general stock, use the GLOBAL KNOWLEDGE. If asked about specific models or specs, use the SPECIFIC INVENTORY DATA.
            2. **Accuracy**: Do not mention technical terms like "RAG," "Vector DB," or "retrieved data." 
            3. **Presentation**: Use bold text for model names. Use structured tables for comparing 3 or more devices.
            4. **List Requirement**: When a user asks about a specific brand's mobiles, ALWAYS present the list of models in a clean bullet-point format.
            5. **Pricing**: Always mention prices in PKR.
            6. **Integrity**: If a user asks for something we don't carry, check the GLOBAL KNOWLEDGE brands and suggest the best alternative from those brands using the SEARCH DATA.
            7. **Tone**: Helpful, elite, and precise. Like a premium concierge.
            """
            
            messages_payload = [{'role': 'system', 'content': system_prompt}] + st.session_state["messages"]
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_payload,
                stream=True,
            )

            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    # Character-by-character for a "softer" and more readable flow
                    for char in content:
                        full_response += char
                        display_html = f'<div class="bot-container">{bot_icon}<div class="bot-content">{full_response}<span class="cursor"></span></div></div>'
                        response_placeholder.markdown(display_html, unsafe_allow_html=True)
                        time.sleep(0.01) # Tiny delay for softness
            
            # Final output without cursor
            display_html = f'<div class="bot-container">{bot_icon}<div class="bot-content">{full_response}</div></div>'
            response_placeholder.markdown(display_html, unsafe_allow_html=True)
            st.session_state["messages"].append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_html = f'<div class="bot-container">{bot_icon}<div class="bot-content"><div style="background-color: #fff5f5; color: #d32f2f; padding: 12px 18px; border-radius: 12px; font-size: 0.95rem; border: 1px solid #ffcdd2;">I encountered a connection issue. Please try again.</div></div></div>'
            response_placeholder.markdown(error_html, unsafe_allow_html=True)
