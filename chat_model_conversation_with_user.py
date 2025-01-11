from dotenv import load_dotenv
from os import getenv

from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

model = ChatOpenAI(model=getenv("SMART_LLM"), api_key=getenv("OPENAI_API_KEY"))

chat_history = []

system_message = SystemMessage(content="You are helpful AI assistant.")
chat_history.append(system_message)

while True:
    query = input("You:\n")
    print(query.lower())
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"AI:\n{response}")

print(f"\n\nMessage History\n{chat_history}")