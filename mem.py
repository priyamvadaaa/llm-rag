import getpass
import os
from dotenv import load_dotenv
from langchain.chains.question_answering.map_reduce_prompt import messages
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

from langchain_google_genai import ChatGoogleGenerativeAI

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt_msg=ChatPromptTemplate.from_messages(
    [
        ("system","you are a very good comedian"),
        ("user","tell me a good joke"),
    ]
)
prompt=prompt_msg.invoke({})
ai_msg=llm.invoke(prompt)
print(ai_msg.content)