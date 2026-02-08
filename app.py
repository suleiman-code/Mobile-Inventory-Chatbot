# import streamlit as st
# import pandas as pd
# import ollama
# import os
# import re

# # Configuration
# MODEL_NAME = "qwen2.5:0.5b"
# CSV_FILE = "MobileInventory.csv"

# # Connect explicitly to IPv4 to avoid localhost DNS lag
# client = ollama.Client(host='http://127.0.0.1:11434')

# def load_data():
#     if os.path.exists(CSV_FILE):
#         try:
#             return pd.read_csv(CSV_FILE)
#         except Exception as e:
#             st.error(f"Error reading CSV file: {e}")
#             return None
#     return None

# def search_data(df, query):
#     if df is None or df.empty:
#         return ""
    
#     # Split query into terms and escape them to handle special chars like '?' safely
#     query_terms = [re.escape(term) for term in query.lower().split()]
    
#     if not query_terms:
#          return ""

#     # Filter rows that contain any of the query terms (naive search)
#     mask = df.apply(lambda row: row.astype(str).str.contains('|'.join(query_terms), case=False).any(), axis=1)
#     relevant_rows = df[mask]
    
#     # Limit to top 3 rows for speed
#     if len(relevant_rows) > 3:
#         relevant_rows = relevant_rows.head(3)
        
#     if relevant_rows.empty:
#         return ""
        
#     return relevant_rows.to_markdown(index=False)

# st.set_page_config(page_title="Mobile Inventory Chatbot", layout="wide")
# st.title("📱 SLM Mobile Inventory Chatbot")

# # Load Data
# df = load_data()

# with st.sidebar:
#     st.header("Data Source")
#     if df is not None:
#         st.success(f"Loaded '{CSV_FILE}'")
#         st.write(f"Total Records: {len(df)}")
#         try:
#             st.dataframe(df.head(5)) 
#         except:
#              st.dataframe(df.head(5))
#     else:
#         st.error(f"File '{CSV_FILE}' not found.")

# # Chat handling
# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "assistant", "content": "👋 Welcome to SLM Mobile World! I'm here to help you find the best phones. What are you looking for today?"}]

# for msg in st.session_state["messages"]:
#     if msg["role"] == "user":
#         st.chat_message("user").write(msg["content"])
#     else:
#         st.chat_message("assistant").write(msg["content"])

# if prompt := st.chat_input("Ask about inventory (e.g., 'Do we have iPhone 15?'):"):
#     st.session_state["messages"].append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     # 1. Search for relevant data
#     context_data = ""
#     if df is not None:
#         context_data = search_data(df, prompt)

#     # 2. Construct Prompt
#     if not context_data:
#         system_prompt = f"User asked: '{prompt}'. No exact matches found. politely apologize, and ask if they would like to see other similar brands like Samsung or iPhone."
#     else:
#         # Friendly but efficient salesperson persona
#         system_prompt = f"""You are a friendly and enthusiastic sales assistant at a mobile shop.
        
#         Data:
#         {context_data}
        
#         Question: {prompt}
        
#         INSTRUCTION: Answer warmly. Highlight the best features if mentioned in data. Keep it concise (2-3 sentences max) so the customer doesn't wait long.
#         """

#     # Debugging helper
#     with st.expander("Debug Info"):
#         st.text(f"Model: {MODEL_NAME}")
#         if context_data:
#             st.code(context_data)
#         else:
#             st.write("No matching rows found.")

#     # 3. Call Model (Non-Streaming for instant-feel feel if fast, or just wait if slow)
#     with st.spinner(f"Checking stock..."):
#         try:
#             full_response = ""
#             # Stream=False generates faster on some backends as it doesn't flush every token
#             response = client.chat(
#                 model=MODEL_NAME,
#                 messages=[{'role': 'system', 'content': system_prompt},
#                           {'role': 'user', 'content': prompt}],
#                 stream=False,
#             )
            
#             full_response = response['message']['content']
            
#             # Save to history
#             st.session_state["messages"].append({"role": "assistant", "content": full_response})
#             st.chat_message("assistant").write(full_response)

#         except Exception as e:
#             st.error(f"Ollama Error: {str(e)}")
#             st.info("Make sure Ollama is running separately in a terminal with 'ollama serve'")
