from getpass import getpass
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from sqlalchemy import create_engine
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()
from langchain.chains import RetrievalQA
#if not os.environ.get("OPENAI_API_KEY"):
#os.environ["OPENAI_API_KEY"] = getpass("Enter API key for OPENAI: ")

connection = "postgresql+psycopg://langchain:langchain@localhost:6029/db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
docs = [
    Document(page_content="there are cats in the pond", metadata={"location": "pond", "topic": "animals"}),
    Document(page_content="ducks are also found in the pond", metadata={"location": "pond", "topic": "animals"}),
    Document(page_content="fresh apples are available at the market", metadata={"location": "market", "topic": "food"}),
    Document(page_content="oranges are available at the market", metadata={"location": "market", "topic": "food"})
]
id=["1","2","3","4"]
vector_store = PGVector(
    embeddings=embeddings,
    collection_name="langchain_pg_embedding",
    connection=connection,
    use_jsonb=True,
)

# vector_store.add_documents(documents=docs,ids=id)
''' manual retrivel directly from postgres
result=vector_store.similarity_search(query="market")
'''
#context="\n".join([f"- {d.page_content}" for d in result])
prompt=ChatPromptTemplate.from_template("""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer""")
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
qa=RetrievalQA.from_llm(
    llm,
    retriever=vector_store.as_retriever(),prompt=prompt
)
query="what can be found at pond"
res = qa.invoke({"query": query})
print(res)

