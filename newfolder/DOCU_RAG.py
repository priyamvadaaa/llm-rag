import os.path
from dotenv import load_dotenv
from flask_restful import Resource, Api
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
import os
load_dotenv()
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
from flask import Flask, Blueprint

d_rag=Blueprint("d_rag",__name__)
api=Api(d_rag)

# sudo docker run --name api-con -e POSTGRES_USER=api -e POSTGRES_PASSWORD=api -e POSTGRES_DB=langapi -p 6020:5432 -d pgvector/pgvector:pg16

user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
db = os.getenv("POSTGRES_DB","langapi")
connection = f"postgresql+psycopg://{user}:{password}@postgres:5432/{db}"


# connection="postgresql+psycopg://api:api@localhost:6020/langapi"
# connection="postgresql+psycopg://api:api@postgres:5432/langapi"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=os.getenv("GEMINI_API_KEY"))
prompt=ChatPromptTemplate.from_template("""You are a helpful assistant.Answer the user queries based on the retrieved pieces of context provided.If you don't know the answer, just say that you don't know.
Question:{question}
Context:{context}
Answer""")

# folder_path='/app/uploads/'
folder_path=os.path.join(os.getcwd(),"uploads")
file_pattern="*.txt"


class embFiles(Resource):
    def get(self):
        #finding files from folder
        files=glob.glob(os.path.join(folder_path,file_pattern))
        all_docs=[]       #empty list
        for file in files:         #looping thru all files in folder
            raw_docs=TextLoader(file).load()
            all_docs.extend(raw_docs)

        text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_split.split_documents(all_docs)


        vs=PGVector(
            embeddings=embeddings,
            collection_name="new_api",
            connection=connection,
            use_jsonb=True,
        )
        # vs.add_documents(documents=docs)
        # query="What are some common behaviors of domestic cats"
        query="What did the president say about economy"

        qa_chain = RetrievalQA.from_llm(
            llm,
            retriever=vs.as_retriever(),
            prompt=prompt
        )
        res=qa_chain.invoke({"query":query})
        return res
api.add_resource(embFiles,'/')

