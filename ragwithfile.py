from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



# docker command=docker run --name docucon -e POSTGRES_USER=bag -e POSTGRES_PASSWORD=password -e POSTGRES_DB=newdb -p 6021:5432 -d pgvector/pgvector:pg16
connection = "postgresql+psycopg://bag:password@localhost:6021/newdb"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
raw_docs=TextLoader('state_of_the_union.txt').load()
text_split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=text_split.split_documents(raw_docs)

vs=PGVector(
    embeddings=embeddings,
    collection_name="rag_doc",
    connection=connection,
    use_jsonb=True,
)

#vs.add_documents(documents=docs)
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
prompt=ChatPromptTemplate.from_template("""you are a helpful assistant. Answer the user queries based on the retrieved pieces of context provided.If you don't know the answer, just say that you don't know.
Question:{question}
Context:{context}
Answer""")
qa_chain=RetrievalQA.from_llm(
    llm,
    retriever=vs.as_retriever(),
    prompt=prompt
)
query="Summarize the main priorities mentioned in the speech in 5 lines?"
res=qa_chain.invoke({"query":query})
print(res)
