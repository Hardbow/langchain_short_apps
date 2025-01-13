from dotenv import load_dotenv
from os import getenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model=getenv("SMART_LLM"), api_key=getenv("OPENAI_API_KEY"))

messages = [
    ("system", "You are a film director. You recommend a good topic for a new movie in {film_genre} film genre."),
    ("human", "Tell me four sentences for script about {theme}"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

chains = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chains.invoke({"film_genre": "road movie", "theme": "love story in Lisbon"})

print(result)