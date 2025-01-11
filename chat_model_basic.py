from dotenv import load_dotenv
from os import getenv
from langchain_openai import ChatOpenAI

load_dotenv()
MY_MODEL = getenv("SMART_LLM")

model = ChatOpenAI(model=MY_MODEL, api_key=getenv("OPENAI_API_KEY"))

result = model.invoke("Who is the best movie director?")
print(f"Answer:\n{result.content}")