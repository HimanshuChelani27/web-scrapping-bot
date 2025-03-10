import os
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from sentence_transformers import SentenceTransformer
import pickle
import time
from typing import List, Dict, Tuple
import openai

# 1. Web Scraping Functions
def scrape_with_beautifulsoup(url: str) -> str:
    """Scrape static webpage content using BeautifulSoup"""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        print(f"Error scraping {url} with BeautifulSoup: {e}")
        return ""

def scrape_with_selenium(url: str) -> str:
    """Scrape dynamic webpage content using Selenium"""
    try:
        # Setup Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Initialize the driver
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        # Load the page
        driver.get(url)
        time.sleep(5)  # Allow time for JavaScript to execute
        
        # Get page content
        page_source = driver.page_source
        driver.quit()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        print(f"Error scraping {url} with Selenium: {e}")
        return ""

# 2. Text Processing and Embedding
def process_text(text: str) -> List[str]:
    """Split text into chunks"""
    # Simple sentence splitting (could be enhanced)
    sentences = text.split('. ')
    # Chunk sentences into manageable pieces (max 512 tokens for most embedding models)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 512:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def embed_text(chunks: List[str], model_name: str = "all-mpnet-base-v2") -> np.ndarray:
    """Convert text chunks to embeddings using SentenceTransformer"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

# 3. Vector Database Operations
class VectorDB:
    def __init__(self, dimension: int = 768):
        """Initialize FAISS vector database"""
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        
    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        """Add embeddings to the database"""
        if self.index.ntotal == 0:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            
        # Add embeddings to FAISS
        self.index.add(embeddings.astype('float32'))
        
        # Store corresponding texts
        self.texts.extend(texts)
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[int], List[float]]:
        """Search for similar vectors"""
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return indices[0], distances[0]
    
    def get_text(self, index: int) -> str:
        """Get text for a given index"""
        return self.texts[index]
    
    def save(self, filepath: str):
        """Save the vector database to disk"""
        with open(filepath + "_texts.pkl", "wb") as f:
            pickle.dump(self.texts, f)
        faiss.write_index(self.index, filepath + "_index.faiss")
        
    @classmethod
    def load(cls, filepath: str):
        """Load vector database from disk"""
        db = cls()
        with open(filepath + "_texts.pkl", "rb") as f:
            db.texts = pickle.load(f)
        db.index = faiss.read_index(filepath + "_index.faiss")
        return db

# 4. LLM Integration
def get_answer_from_llm(query: str, context: str, api_key: str = None) -> str:
    """Get answer from LLM using context"""
    if api_key:
        # Using OpenAI API
        openai.api_key = api_key
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer the query based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuery: {query}"}
            ]
        )
        return response.choices[0].message.content
    else:
        # Fallback to a simple approach if no API key
        return f"Based on the context:\n\n{context}\n\nI would answer: The query relates to the information in the context."

# 5. Main Pipeline
class WebScrapingQAPipeline:
    def __init__(self, embedding_model: str = "all-mpnet-base-v2", openai_api_key: str = None):
        """Initialize the pipeline"""
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_db = VectorDB(dimension=self.embedding_model.get_sentence_embedding_dimension())
        self.openai_api_key = openai_api_key
        
    def scrape_and_index(self, urls: List[str], use_selenium: bool = False):
        """Scrape content from URLs and index it"""
        for url in urls:
            # Choose scraping method
            if use_selenium:
                text = scrape_with_selenium(url)
            else:
                text = scrape_with_beautifulsoup(url)
                
            if not text:
                continue
                
            # Process text into chunks
            chunks = process_text(text)
            
            # Get embeddings
            embeddings = self.embedding_model.encode(chunks)
            
            # Add to vector DB
            self.vector_db.add_embeddings(embeddings, chunks)
            
        print(f"Indexed {self.vector_db.index.ntotal} text chunks from {len(urls)} URLs")
            
    def answer_query(self, query: str, top_k: int = 3) -> str:
        """Answer a query using the indexed content"""
        # Embed query
        query_embedding = self.embedding_model.encode([query])
        
        # Retrieve similar contexts
        indices, distances = self.vector_db.search(query_embedding, k=top_k)
        
        # Retrieve corresponding texts
        contexts = [self.vector_db.get_text(idx) for idx in indices]
        combined_context = "\n\n".join(contexts)
        
        # Get answer from LLM
        answer = get_answer_from_llm(query, combined_context, self.openai_api_key)
        
        return answer
    
    def save(self, directory: str = "qa_pipeline"):
        """Save pipeline to disk"""
        os.makedirs(directory, exist_ok=True)
        self.vector_db.save(os.path.join(directory, "vector_db"))
        
    @classmethod
    def load(cls, directory: str = "qa_pipeline", embedding_model: str = "all-mpnet-base-v2", openai_api_key: str = None):
        """Load pipeline from disk"""
        pipeline = cls(embedding_model, openai_api_key)
        pipeline.vector_db = VectorDB.load(os.path.join(directory, "vector_db"))
        return pipeline

# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = WebScrapingQAPipeline(openai_api_key="your-openai-api-key")
    
    # Example URLs to scrape
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    # Scrape and index content
    pipeline.scrape_and_index(urls, use_selenium=False)
    
    # Save the pipeline
    pipeline.save()
    
    # Example query
    query = "What are the main approaches to machine learning?"
    answer = pipeline.answer_query(query)
    print(f"Query: {query}")
    print(f"Answer: {answer}")
    
    # Later, you can load the pipeline
    # loaded_pipeline = WebScrapingQAPipeline.load(openai_api_key="your-openai-api-key")
    # answer = loaded_pipeline.answer_query("What is deep learning?")
