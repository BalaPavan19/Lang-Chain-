from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.chains import RetrievalQA,LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnableMap
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent,initialize_agent,AgentType
from langchain.schema import Document
import psycopg2
import requests
from util import llm


#========================================================================================
# RAG
#========================================================================================
# 1. Load -------------------------------------------------------------------------------------------

# load Text
def text_loader():
    return TextLoader("data.txt", encoding="utf-8")

# load PDF
def pdf_loader():
    return PyPDFLoader("Two_Pages.pdf")

#load WEbPage
def web_loader():
    return WebBaseLoader("https://github.com/krishnaik06/RAG-Tutorials/blob/main/notebook/document.ipynb")


# Load North wind Odata
def odata_loader():
    url = "https://services.odata.org/V4/Northwind/Northwind.svc/Products?$format=json"
    response = requests.get(url)
    data = response.json()

    documents = []
    for item in data["value"]:
        text = "\n".join(f"{key}: {value}" for key, value in item.items())
        # documents.append(Document(page_content=text, metadata={"ProductID": item.get("ProductID")}))   #Metadata is additional info used for filtering
        documents.append(Document(page_content=text ))
    return documents

# load Postgres
def load_from_postgres():
    conn = psycopg2.connect(
        dbname="dbname",
        user="username",
        password="password",
        host="abcdefgh.us-east-1.elb.amazonaws.com",
        port="5432"
    )
    cursor = conn.cursor()
    query = """
        SELECT *
        FROM extractiondetails.document_extraction
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    colnames = [desc[0] for desc in cursor.description]
    conn.close()

    documents = []
    for row in rows:
        # Combine all columns into one text blob
        text = "\n".join(f"{col}: {val}" for col, val in zip(colnames, row) if val is not None)
        documents.append(Document(page_content=text, metadata={"id": row[0]}))
    return documents

# documents = odata_loader() 
documents = pdf_loader().load()


# 2. Split into chunks ------------------------------------------------------------------------------------------------
def text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    return text_splitter.split_documents(documents)

 
# 3. Create embeddings and vector store---------------------------------------------------------------------------------
def vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(text_splitter(), embeddings, persist_directory="rag_store")

# vector_store.persist()  #storing locally

# 4. Create QA chain -----------------------------------------------------------------------------------------------
def retrieval_qa_chain():
    retriever = vector_store().as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


def rag_bot():
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        #   docs = retriever.get_relevant_documents(question)        # Gets docs from vectorDB through given query
        #   for i, doc in enumerate(docs):
        #     print(f"\nDocument {i+1}:\n{doc.page_content}")
        if question.lower() == "exit":
            break
        answer = retrieval_qa_chain().run(question)
        print(f"\nAnswer: {answer}")


rag_bot()
