from dotenv import load_dotenv
from os import getenv

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model=getenv("SMART_LLM"), api_key=getenv("OPENAI_API_KEY"))

print("------------Prompt from Template-----------------")
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({"topic": "cats"})
result = model.invoke(prompt)
print(result.content)

print("\n------------Prompt with multiple Placeholders-------------")
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}.
Assistant:"""
prompt_template = ChatPromptTemplate.from_template(template_multiple)

prompt = prompt_template.invoke({"adjective": "funny", "animal": "panda"})
result = model.invoke(prompt)
print(result.content)

print("\n------------Prompt with System and Human Messages (Tuple)-------------")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print(result.content)

