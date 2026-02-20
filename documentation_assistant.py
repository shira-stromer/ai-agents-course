from dotenv import load_dotenv
import logging
import os
import ssl 
from typing import Any 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeSparseVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
import certifi

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
logger.addHandler(handler)

load_dotenv()  # Load environment variables from .env file

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10)

vectore_store = Chroma(collection_name="documentation-assistant", embedding_function=embeddings, persist_directory="chroma_db")
#vectore_store = PineconeSparseVectorStore(index_name="documentation-assistant", embedding_function=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

def main():
    logger.info("Documentation Assistant started.")

    logger.info("Crawling documentation from langchain.com...")
    
    res = tavily_crawl.invoke({
        "url": "https://docs.langchain.com/oss/python/langchain/overview",
        "max_depth": 1, #5
        "extract_depth": "advanced",
        "instructions": "content on ai agents"
    })

    all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]
    logger.info(f"Crawled {len(all_docs)} documents.")
    pass

if __name__ == "__main__":
    main()