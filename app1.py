import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import re

# Configuration
MODEL_NAME = "gpt-3.5-turbo"
CSV_FILE = "MobileInventory.csv"

# Initialize OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Fallback to streamlit secrets if available, or show error
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        st.error("OPENAI_API_KEY not found. Please set it as an environment variable or in .streamlit/secrets.toml")
        st.stop()

client = OpenAI(api_key=api_key)

st.set_page_config(page_title="SLM Mobile AI", layout="wide", page_icon="📱")

# Custom Premium Styling (Light/Grey Theme for visibility)
st.markdown("""
    <style>
    /* Premium Grey Theme */
    :root {
        --bg-color: #ffffff;
        --sidebar-bg: #f8f9fa;
        --text-main: #1a1a1a;
        --text-secondary: #444444;
        --accent-blue: #00d2ff;
        --border-color: #e0e0e0;
    }

    .main {
        background-color: var(--bg-color);
        color: var(--text-main);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-main) !important;
    }

    .stChatMessage {
        background-color: transparent !important;
        border-bottom: 1px solid #f0f0f0 !important;
        padding: 1.5rem 0 !important;
        max-width: 800px;
        margin: 0 auto !important;
    }

    /* Message content styling */
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        color: var(--text-main) !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Input area styling */
    .stChatInputContainer {
        background-color: white !important;
        padding-bottom: 20px;
    }
    
    .stChatInputContainer textarea {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #cccccc !important;
        border-radius: 12px !important;
    }

    h1, h2, h3 {
        color: #1a1a1a !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }

    /* Hide streamlit clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stMetric {
        background: white;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    [data-testid="stMetricLabel"] {
        color: #666666 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2rem;
        padding: 10px;
    }
    
    .logo-text {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

def load_data():
    if os.path.exists(CSV_FILE):
        try:
            return pd.read_csv(CSV_FILE)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    return None

def search_data(df, query):
    if df is None or df.empty:
        return ""
    
    query = query.lower()
    brands = [b.lower() for b in df['Brand'].unique()]
    
    # Check if a specific brand is the main subject
    queried_brand = None
    for brand in brands:
        if brand in query:
            queried_brand = brand
            break
            
    if queried_brand:
        # If brand is mentioned, get ALL models for that brand
        relevant_rows = df[df['Brand'].str.lower() == queried_brand]
    else:
        # General search with higher limit
        query_terms = [re.escape(term) for term in query.split()]
        if not query_terms:
             return ""
        mask = df.apply(lambda row: row.astype(str).str.contains('|'.join(query_terms), case=False).any(), axis=1)
        relevant_rows = df[mask]
        # Increase limit to 15 for better context
        if len(relevant_rows) > 15:
            relevant_rows = relevant_rows.head(15)
        
    if relevant_rows.empty:
        return ""
        
    return relevant_rows.to_markdown(index=False)

# Dashboard Logic
df = load_data()

with st.sidebar:
    # Adding Logo
    st.markdown("""
        <div class="logo-container">
            <img src="https://img.icons8.com/ios-filled/50/4a90e2/iphone-x.png" width="35">
            <span class="logo-text">SLM Mobile AI</span>
        </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        st.subheader("Inventory Status")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Total Models", len(df))
        with m2:
            st.metric("Unique Brands", df['Brand'].nunique())
            
        st.divider()
        st.metric("Total Stock", df['Stock'].sum())

        st.divider()
        st.caption("Active Inventory v4.2")
    else:
        st.error(f"File '{CSV_FILE}' not found.")

# Chat initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "👋 Welcome to SLM Mobile AI. I'm your premium assistant for finding the perfect mobile device. What can I do for you today?"}]

# Centered container for chat
st.markdown("""
    <div style="text-align: center; padding: 20px 0 40px 0;">
        <h1 style="font-size: 3rem; font-weight: 800; color: #1a1a1a; margin-bottom: 5px;">
            📱 SLM <span style="color: #00d2ff;">Mobile</span> Inventory
        </h1>
        <p style="color: #666; font-size: 1.1rem; letter-spacing: 1px;">PREMIUM AI ASSISTANT</p>
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

    # 1. Search for relevant data
    context_data = ""
    global_metadata = ""
    if df is not None:
        context_data = search_data(df, prompt)
        # Prepare a summary of available brands and inventory
        brands = df['Brand'].unique().tolist()
        total_items = len(df)
        global_metadata = f"Available Brands: {', '.join(brands)}. Total unique models in stock: {total_items}."

    # 2. Construct Prompt
    system_prompt = f"""You are the **Lead Sales Concierge** at SLM Mobile World. Your goal is to provide a premium, white-glove experience.

    You have access to a full inventory of {total_items} products. When a user asks for a specific brand (e.g., "Apple" or "Samsung"), you MUST list ALL models for that brand found in the context below. Do not omit any items.

    **GLOBAL INVENTORY OVERVIEW:**
    {global_metadata}
    
    **SPECIFIC MODELS FOUND IN DATABASE:**
    {context_data if context_data else "No specific models matched your prompt in our database."}

    **STRICT RESPONSE PROTOCOL:**
    1. **Complete Listing**: If a brand is requested, display EVERY model for that brand provided in the context.
    2. **Premium Style**: Use bold text for emphasis and bullet points for detailed specs.
    3. **Pricing Requirement**: You MUST include the price in PKR for every device mentioned.
    4. **No Hallucination**: Only talk about phones listed in the 'SPECIFIC MODELS FOUND' section.
    5. **Markdown Tables**: For lists of 4 or more models, use a Markdown table (Columns: Model Name, RAM, Storage, Price PKR).
    6. **Closing**: End with a helpful follow-up question.
    """

    # Debugging helper
    with st.expander("Debug Info"):
        st.text(f"Model: {MODEL_NAME}")
        if context_data:
            st.code(context_data)
        else:
            st.write("No matching rows found.")

    # 3. Call Model (Non-Streaming for instant-feel feel if fast, or just wait if slow)
    with st.spinner(f"Checking stock..."):
        try:
            full_response = ""
            # Stream=False generates faster on some backends as it doesn't flush every token
            # Include full conversation history for memory
            messages_payload = [{'role': 'system', 'content': system_prompt}] + st.session_state["messages"]
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_payload,
                stream=False,
            )
            
            full_response = response.choices[0].message.content
            
            # Save to history
            st.session_state["messages"].append({"role": "assistant", "content": full_response})
            st.chat_message("assistant").write(full_response)

        except Exception as e:
            st.error(f"OpenAI Error: {str(e)}")
            st.info("Please check your API key and internet connection.")
