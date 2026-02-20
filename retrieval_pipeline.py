from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

print("Initializing components...")

embeddings = OpenAIEmbeddings()

vector_store = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"],
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

def retrieval_chain_with_lcel(query: str):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content

prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the following context: 
    {context}

    Question: {question}

    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format the retrieved documents into a string."""
    return "\n\n".join([doc.page_content for doc in docs])

def create_retrieval_chain_with_lcel():
    return (RunnablePassthrough
            .assign(context=itemgetter("question") | retriever | format_docs)
            | prompt_template | llm | StrOutputParser())

if __name__ == "__main__":
    print("Retrieving...")

    query = "What is Pinecone in Machine Learning?"

    print("retrieving with LLM and vector store without LCEL...")
    print(retrieval_chain_with_lcel(query))

    print("retrieving with LLM and vector store with LCEL...")
    retrieval_chain = create_retrieval_chain_with_lcel()
    print(retrieval_chain.invoke({"question": query}))