# 📖 PDF Q&A Chatbot using Streamlit & Hugging Face API

## 🚀 Overview
This project is a **PDF-based Q&A chatbot** built with **Streamlit**, **FAISS** for similarity search, and **Hugging Face Inference API** for generating answers. Users can upload a **PDF file**, ask questions about its content, and receive answers using an **LLM (e.g., Mistral-7B, Falcon-7B, Llama-3, etc.)**.

---

## 🎯 Features
✅ **Upload PDFs** & extract text + tables 📄  
✅ **Vector search with FAISS** for retrieving relevant text 🔍  
✅ **Generate answers using Hugging Face Inference API** 🤖  
✅ **Streamlit UI for easy interaction** 🖥️  
✅ **Supports any Hugging Face text generation model** 🔄  

---

## 🛠️ Installation

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/ShubhamMandowara/llm_rag.git
cd pdf-qna-chatbot
```

### **2️⃣ Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🔑 Set Up Hugging Face API Token
1. Get your API token from [Hugging Face Tokens](https://huggingface.co/settings/tokens).
2. When running the Streamlit app, enter the API token in the provided input field.
3. Or, store it in **Streamlit secrets** by adding to `.streamlit/secrets.toml`:
   ```toml
   [secrets]
   HUGGINGFACEHUB_API_TOKEN = "your_token_here"
   ```

---

## 🎯 Usage

### **Run the Streamlit App**
```bash
streamlit run app.py
```

### **How It Works**
1️⃣ **Upload a PDF** to extract text & tables.  
2️⃣ **FAISS retrieves relevant text chunks** from the document.  
3️⃣ **Hugging Face Inference API** generates an answer based on context.  
4️⃣ **Streamlit UI displays the answer.** 🎉  

---

## 🏗️ Project Structure
```
📂 pdf-qna-chatbot
│── app.py              # Main Streamlit app
│── requirements.txt    # Required dependencies
│── secrets.toml         # Streamlit config folder (for secrets)
```

---

## 📌 Example Models (Change in `app.py`)
- **Mistral-7B**: `mistralai/Mistral-7B-Instruct`
- **Llama-3**: `meta-llama/Llama-3-8B`
- **Falcon-7B**: `tiiuae/falcon-7b-instruct`

To change the model, update:
```python
HF_MODEL = "mistralai/Mistral-7B-Instruct"  # Change to your preferred model
```

---

## 🛠️ Troubleshooting
### **1️⃣ Model Not Found / API Issues**
- Ensure your **Hugging Face API token** is correct.
- Check if the model is **public** and supports the **Inference API**.
- Try testing API manually:
  ```bash
  curl -H "Authorization: Bearer your_api_token" https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct
  ```

### **2️⃣ FAISS Index Not Working**
- Ensure FAISS is installed:
  ```bash
  pip install faiss-cpu  # Or faiss-gpu if using CUDA
  ```

### **3️⃣ Streamlit Not Running**
- Restart the app after setting environment variables:
  ```bash
  streamlit run app.py
  ```

---

## 📜 License
This project is **open-source** under the GPL-3.0 License.

---

## 👨‍💻 Author
Developed by [Shubham Mandowara](https://github.com/shubhammandowara). Contributions are welcome! 😊🚀