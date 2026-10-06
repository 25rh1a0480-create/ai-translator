import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from PIL import Image
import ollama

# 1. Page Title & Banner Image (Pillow)
st.title("AI Text & PDF Translator")

# Load a local header/banner image using Pillow if needed
banner_img = Image.new("RGB", (600, 50), color=(40, 100, 150))
st.image(banner_img, use_container_width=True)

# 2. Setup Vector Database (ChromaDB + SentenceTransformers)
@st.cache_resource
def setup_database():
    client = chromadb.Client()
    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name="translator_db", 
        embedding_function=embedding_model
    )

db = setup_database()

# 3. User Controls
language = st.selectbox("Select Target Language:", ["Spanish", "French", "German", "Hindi", "Japanese"])

# PDF Upload option (PyPDF)
uploaded_file = st.file_uploader("Upload a PDF (Optional Reference Context)", type=["pdf"])

pdf_text = ""
if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text + " "

user_text = st.text_area("Enter Text to Translate:", height=150)

# Combine manual input and PDF input
combined_text = (user_text + " " + pdf_text).strip()

# 4. Translation Action
if st.button("Translate Text"):
    if combined_text:
        # Step A: Split text into chunks/sentences
        sentences = [s.strip() for s in combined_text.split(".") if s.strip()]
        ids = [f"id_{i}" for i in range(len(sentences))]
        
        # Step B: Store text in ChromaDB
        db.upsert(ids=ids, documents=sentences)
        
        # Step C: Retrieve matching content from database
        results = db.query(query_texts=[combined_text], n_results=min(5, len(sentences)))
        retrieved_text = " ".join(results["documents"][0])
        
        # Step D: Send prompt to Ollama LLM
        prompt = f"Translate the following text into {language}. Only return the translation:\n\n{retrieved_text}"
        
        with st.spinner("Translating..."):
            try:
                response = ollama.generate(model="llama3", prompt=prompt)
                st.subheader(f"Translation ({language}):")
                st.write(response["response"])
            except Exception as e:
                st.error(f"Ollama Error: Make sure Ollama is running. ({e})")
    else:
        st.warning("Please enter text or upload a PDF first!")