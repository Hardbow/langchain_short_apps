from dotenv import load_dotenv
from os import getenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model=getenv("SMART_LLM"), api_key=getenv("OPENAI_API_KEY"))

messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

chain = prompt_template | model | StrOutputParser()

result = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result)