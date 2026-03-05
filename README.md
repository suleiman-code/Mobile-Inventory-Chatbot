# 📱 SLM Mobile Inventory AI

A premium AI-powered inventory management and sales assistant built with Streamlit and OpenAI. This application allows users to interact with a mobile phone inventory of 500+ products using a natural language interface, similar to ChatGPT and Google Gemini.

## ✨ Features

- **Gemini & ChatGPT-inspired Interface**: Sleek, modern professional theme with centered layouts and premium typography.
- **Smart RAG Engine**: Retrieval-Augmented Generation using LangChain and FAISS for accurate inventory search.
- **Context Memory**: Remembers previous conversation turns for a natural, multi-turn chat experience.
- **Streaming Responses**: Real-time word-by-word response generation (Typewriter effect).
- **Custom Branding**: Fully customized UI with mobile logos, custom favicons, and branded elements.
- **Large Inventory**: Comprehensive database (`MobileInventory.csv`) containing 500+ products across 10 top brands.
- **Professional Presentation**: Automatically generates tables for comparisons and provides pricing in PKR.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **RAG & AI Logic**: [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss)
- **AI Engine**: [OpenAI GPT-3.5 Turbo](https://openai.com/)
- **Data Handling**: [Pandas](https://pandas.pydata.org/)
- **Styling**: Vanilla CSS for premium customizations.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/suleiman-code/Mobile-Inventory-Chatbot.git
   cd Mobile-Inventory-Chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API Key:
   Create a `.streamlit/secrets.toml` file:
   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   ```

4. Run the application:
   ```bash
   python -m streamlit run app1.py
   ```

## 📁 Project Structure

- `app1.py`: Main application code with custom styling, RAG logic, and memory management.
- `MobileInventory.csv`: Database containing 500 mobile phone records.
- `requirements.txt`: Python dependencies.
- `.streamlit/`: Streamlit configuration and secrets.

---
Developed by **SLM Mobile World**
