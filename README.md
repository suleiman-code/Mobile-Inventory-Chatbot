# 📱 SLM Mobile Inventory AI

A premium AI-powered inventory management and sales assistant built with Streamlit and OpenAI. This application allows users to interact with a mobile phone inventory of 100+ products using a natural language interface, similar to ChatGPT.

## ✨ Features

- **ChatGPT-like Interface**: Centered chat layout with a sleek, modern professional theme.
- **Smart Inventory Search**: Advanced search logic that handles specific brand queries and general product searches.
- **Real-time Metrics**: Sidebar dashboard showing total models, unique brands, and total stock.
- **Rich Data**: Comprehensive database (`MobileInventory.csv`) containing 100 products from top brands like Apple, Samsung, Google, OnePlus, Xiaomi, Vivo, Oppo, and Realme.
- **Intelligent Response Protocol**: Automated table generation for comparisons and detailed spec listings with mandatory PKR pricing.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **AI Engine**: [OpenAI GPT-3.5 Turbo](https://openai.com/)
- **Data Handling**: [Pandas](https://pandas.pydata.org/)
- **Styling**: Custom CSS for a premium white-glove experience.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Slm_MobileInventory.git
   cd Slm_MobileInventory
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API Key:
   Create a `.streamlit/secrets.toml` file or set an environment variable:
   ```toml
   OPENAI_API_KEY = "your_api_key_here"
   ```

4. Run the application:
   ```bash
   python -m streamlit run app1.py
   ```

## 📁 Project Structure

- `app1.py`: Main application code with custom styling and AI logic.
- `MobileInventory.csv`: Database containing 100 mobile phone records.
- `requirements.txt`: Python dependencies.
- `.streamlit/`: Streamlit configuration and secrets.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
Developed by **SLM Mobile World**
