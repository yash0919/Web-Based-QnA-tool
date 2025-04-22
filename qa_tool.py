import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Set page configuration as the first command
st.set_page_config(page_title="Web-based Q&A Tool", layout="wide")

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to scrape content from a URL
def scrape_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return None

# Function to embed text and create a FAISS index
def create_faiss_index(texts):
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance
    index.add(embeddings)
    return index

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f0f2f5;
    }
    .title {
        font-size: 2.5em;
        color: #4a4a4a;
        text-align: center;
    }
    .subheader {
        font-size: 1.5em;
        color: #6c757d;
        text-align: center;
    }
    .stTextInput>div>div>input {
        border: 2px solid #007bff;
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        border: 2px solid #007bff;
        border-radius: 5px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<h1 class="title">Web-based Q&A Tool</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subheader">Extract content from webpages and ask questions based on that content.</h2>', unsafe_allow_html=True)

# Sidebar for URL input
st.sidebar.header("Input URLs")
urls = st.sidebar.text_area("Enter URLs (one per line):", height=150)

if st.sidebar.button("Ingest Content"):
    if urls:
        url_list = urls.splitlines()
        all_content = []
        with st.spinner("Ingesting content..."):
            for url in url_list:
                content = scrape_content(url)
                if content:
                    all_content.append(content)

        if all_content:
            # Create FAISS index
            index = create_faiss_index(all_content)
            st.session_state['index'] = index
            st.session_state['content'] = all_content
            st.success("Content ingested successfully!")
        else:
            st.error("No valid content was ingested.")
    else:
        st.warning("Please enter at least one URL.")

# Input for questions
if 'index' in st.session_state:
    st.sidebar.header("Ask a Question")
    question = st.sidebar.text_input("Type your question:")
    if st.sidebar.button("Get Answer"):
        if question:
            with st.spinner("Searching for an answer..."):
                # Embed the question
                question_embedding = model.encode([question])
                
                # Search for the closest content
                D, I = st.session_state['index'].search(question_embedding, k=1)  # Get the top 1 result
                if I[0][0] != -1:  # Check if a valid index is returned
                    answer = st.session_state['content'][I[0][0]]
                    st.write("### Answer:")
                    st.write(answer)
                else:
                    st.write("No relevant content found.")
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please ingest content first by entering URLs.")