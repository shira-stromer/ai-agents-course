import os
from dotenv import load_dotenv
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == "__main__":
    print("Ingesting data...")
    loader = TextLoader(file_path="medium_blog.txt")
    document = loader.load()

    print("Splitting data into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks of data.")

    embeddings = OpenAIEmbeddings()

    print("Ingesting")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
    pass