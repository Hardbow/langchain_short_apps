from dotenv import load_dotenv
from os import getenv

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model=getenv("SMART_LLM"), api_key=getenv("OPENAI_API_KEY"))

messages = [
    ("system", "You are a film director. You recommend good topic for a new movie in {film_genre} film genre"),
    ("human", "Tell me short information for script about {theme}")
]

prompt_template = ChatPromptTemplate.from_messages(messages)

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

response = chain.invoke({"film_genre": "road movie", "theme": "love story in Lisbon"})
print(response)